import numpy as np
import cv2
import threading
import time

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

import HostMotorController.common as common
from HostMotorController.motor import SerialMotorController

from PID import PID

try:
    from cv2 import legacy
    MOSSE_AVAILABLE = True
except ImportError:
    MOSSE_AVAILABLE = False

FOV_X = np.deg2rad(50)
FOV_Y = np.deg2rad(34)

PAN_RATE_MAX = np.deg2rad(180)
TILT_RATE_MAX = np.deg2rad(180)
MOTOR_SMOOTHING = 0.25
TRACKING_GAIN = 3.0
fps_smoothing = 0.9

class RocketTracker:
    def __init__(self, model, encoder, motor_port, motor_baud):
        self.model = model
        self.encoder = encoder

        self.motors = SerialMotorController(motor_port, motor_baud)
        self.motors.run()

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
        self.tracker = Tracker(self.metric)

        self.pan_pid = PID(2.4, 0.08, 0.18, integral_limit=np.deg2rad(12), output_limit=np.deg2rad(120))
        self.tilt_pid = PID(2.4, 0.08, 0.18, integral_limit=np.deg2rad(12), output_limit=np.deg2rad(120))
        self.pan, self.tilt = 0.0, 0.0

        self.id_to_track = None
        self.filtered_pan_freq = 0.0
        self.filtered_tilt_freq = 0.0
        
        # MOSSE tracker for intermediate frames
        self.mosse_trackers = {}  # Maps track_id to (mosse_tracker, last_bbox)
        self.last_yolo_frame = None

    def _command_motors(self, pan_rate, tilt_rate):
        # Map angular rate (rad/s) to board frequency command.
        x_freq = (pan_rate / PAN_RATE_MAX) * common.MAX_FREQ
        y_freq = (tilt_rate / TILT_RATE_MAX) * common.MAX_FREQ

        if abs(x_freq) < 1.0:
            x_freq = 0.0
        if abs(y_freq) < 1.0:
            y_freq = 0.0

        self.filtered_pan_freq = MOTOR_SMOOTHING * x_freq + (1.0 - MOTOR_SMOOTHING) * self.filtered_pan_freq
        self.filtered_tilt_freq = MOTOR_SMOOTHING * y_freq + (1.0 - MOTOR_SMOOTHING) * self.filtered_tilt_freq

        if abs(self.filtered_pan_freq) < 0.5:
            self.filtered_pan_freq = 0.0
        if abs(self.filtered_tilt_freq) < 0.5:
            self.filtered_tilt_freq = 0.0

        self.motors.move(self.filtered_pan_freq, self.filtered_tilt_freq)

    def _initialize_mosse_trackers(self, frame, tracks):
        """Initialize or reinitialize MOSSE trackers based on current tracks."""
        if not MOSSE_AVAILABLE:
            return
        
        current_track_ids = {track.track_id for track in tracks}
        
        # Remove MOSSE trackers for tracks that no longer exist
        self.mosse_trackers = {
            tid: tracker_data for tid, tracker_data in self.mosse_trackers.items()
            if tid in current_track_ids
        }
        
        # Initialize MOSSE trackers for new tracks
        for track in tracks:
            if track.track_id not in self.mosse_trackers and track.time_since_update <= 1:
                try:
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                    bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to OpenCV format (x, y, w, h)
                    
                    mosse = legacy.TrackerMOSSE_create()
                    mosse.init(frame, bbox)
                    self.mosse_trackers[track.track_id] = {
                        'tracker': mosse,
                        'bbox': bbox
                    }
                except Exception as e:
                    print(f"Failed to initialize MOSSE for track {track.track_id}: {e}")

    def _update_mosse_trackers(self, frame):
        """Update MOSSE trackers and return detections."""
        if not MOSSE_AVAILABLE or not self.mosse_trackers:
            return []
        
        mosse_detections = []
        failed_tracks = []
        
        for track_id, tracker_data in self.mosse_trackers.items():
            try:
                success, bbox = tracker_data['tracker'].update(frame)
                
                if success:
                    x, y, w, h = bbox
                    mosse_detections.append({
                        'track_id': track_id,
                        'bbox': [x, y, w, h],
                        'xyxy': [x, y, x + w, y + h]
                    })
                    self.mosse_trackers[track_id]['bbox'] = bbox
                else:
                    failed_tracks.append(track_id)
            except Exception as e:
                print(f"MOSSE update failed for track {track_id}: {e}")
                failed_tracks.append(track_id)
        
        # Remove failed trackers
        for track_id in failed_tracks:
            del self.mosse_trackers[track_id]
        
        return mosse_detections

    def _detections_from_mosse(self, frame, mosse_detections):
        """Convert MOSSE-tracked bboxes to Detection objects for deep_sort."""
        if not mosse_detections:
            return []
        
        bboxes = [det['bbox'] for det in mosse_detections]
        features = self.encoder(frame, bboxes)
        
        detections = [
            Detection(bbox, 0.95, feature)  # Use high confidence for MOSSE tracks
            for bbox, feature in zip(bboxes, features)
        ]
        
        return detections

    def _get_detections(self, frame):
        bboxes = []
        scores = []
        # Classes = [0] for person only
        results = self.model(frame, conf=0.25, device="mps", verbose=False)[0]
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                w = x2 - x1
                h = y2 - y1

                bboxes.append([x1, y1, w, h])
                scores.append(conf)

        if len(bboxes) == 0:
            print("No detections found.")
            return []
        
        features = self.encoder(frame, bboxes)

        detections = [
            Detection(bbox, score, feature)
            for bbox, score, feature in zip(bboxes, scores, features)
        ]

        return detections

    def _prompt_for_tracking_id(self):
        self.id_to_track = int(input("Enter the tracking ID: "))

    def _display_track(self, track, frame):
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_id = track.track_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _align_camera_to_track(self, target_track, frame_shape, dt):
        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        x1, y1, x2, y2 = target_track.to_tlbr()
        target_x = (x1 + x2) / 2.0
        target_y = (y1 + y2) / 2.0

        err_x = (target_x - frame_center_x) / (frame_width / 2)
        err_y = (frame_center_y - target_y) / (frame_height / 2)
        err_x = np.clip(err_x, -1.0, 1.0) * TRACKING_GAIN
        err_y = np.clip(err_y, -1.0, 1.0) * TRACKING_GAIN 

        # Ignore tiny errors near image center to reduce jitter.
        deadzone = 0.05
        if abs(err_x) < deadzone:
            err_x = 0.0
        if abs(err_y) < deadzone:
            err_y = 0.0

        # Convert to angle error
        err_pan = err_x * (FOV_X / 2)
        err_tilt = err_y * (FOV_Y / 2)

        # PID control
        pan_rate = self.pan_pid.update(err_pan, dt)
        tilt_rate = self.tilt_pid.update(err_tilt, dt)
        self.pan += pan_rate * dt
        self.tilt += tilt_rate * dt
        self._command_motors(pan_rate, tilt_rate)

    def run(self, camera_index=0):
        if self.motors is None:
            return

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        fps = 0.0

        ret, frame1 = cap.read()
        if not ret:
            return
        
        detections = self._get_detections(frame1)

        if len(detections) == 0:
            return

        self.tracker.predict()
        self.tracker.update(detections)

        for track in self.tracker.tracks:
            if track.time_since_update > 1:
                continue

            self._display_track(track, frame1)

        prompt_thread = threading.Thread(target=self._prompt_for_tracking_id, daemon=True)
        prompt_thread.start()

        cv2.imshow("Select Target", frame1)
        while prompt_thread.is_alive():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        prompt_thread.join()
        
        cv2.destroyAllWindows()

        print(f"Selected tracking ID: {self.id_to_track}")

        frames_to_skip = 5
        frame_count = 0

        last_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            current_fps = 1.0 / dt
            fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            self.tracker.predict()
            frame_count += 1
            
            # Use YOLO on regular intervals, MOSSE on skipped frames
            if frame_count % frames_to_skip == 0:
                detections = self._get_detections(frame)
                self.tracker.update(detections)
                
                # Initialize/update MOSSE trackers based on YOLO detections
                self._initialize_mosse_trackers(frame, self.tracker.tracks)
                frame_count = 0
            else:
                # Use MOSSE tracker on intermediate frames
                mosse_detections = self._update_mosse_trackers(frame)
                if mosse_detections:
                    mosse_detections_obj = self._detections_from_mosse(frame, mosse_detections)
                    self.tracker.update(mosse_detections_obj)

            target_exists = False
            for track in self.tracker.tracks:
                if track.track_id != self.id_to_track:
                    continue
                
                target_exists = True
                self._display_track(track, frame)
                self._align_camera_to_track(track, frame.shape, dt)
                    
            if not target_exists:
                self.motors.move(0, 0)

            cv2.imshow("Rocket Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.motors.move(0, 0)
        self.motors.close()
        cv2.destroyAllWindows()