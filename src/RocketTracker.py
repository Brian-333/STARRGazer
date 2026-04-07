import numpy as np
import cv2
import threading
import time
from collections import deque

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

import HostMotorController.common as common
from HostMotorController.motor import SerialMotorController

from PID import PID

FOV_X = np.deg2rad(50)
FOV_Y = np.deg2rad(34)

PAN_RATE_MAX = np.deg2rad(180)
TILT_RATE_MAX = np.deg2rad(180)
MOTOR_SMOOTHING = 0.25
TRACKING_GAIN = 3.0
fps_smoothing = 0.9

class SimpleBBox:
    """Wrapper for MOSSE bounding box to match track interface."""
    def __init__(self, bbox):
        self.bbox = bbox
    
    def to_tlbr(self):
        """Return bounding box in top-left, bottom-right format."""
        return self.bbox

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
        
        # MOSSE tracker for efficient frame-to-frame tracking
        self.mosse = None
        self.mosse_bbox = None
        
        # FPS tracking
        self.fps_history = deque(maxlen=100)  # Keep last 100 FPS values

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

    def _init_mosse(self, frame, bbox):
        """Initialize MOSSE tracker with initial bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        self.mosse = cv2.legacy.TrackerMOSSE_create()
        self.mosse.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.mosse_bbox = bbox

    def _update_mosse(self, frame):
        """Update MOSSE tracker and return new bounding box."""
        if self.mosse is None:
            return None
        
        success, bbox = self.mosse.update(frame)
        if success:
            x, y, w, h = bbox
            self.mosse_bbox = (x, y, x + w, y + h)
            return self.mosse_bbox
        return None

    def _draw_fps_plot(self, frame, fps):
        """Draw FPS history plot on the frame."""
        self.fps_history.append(fps)
        
        # Plot dimensions
        plot_width = 200
        plot_height = 100
        plot_x = frame.shape[1] - plot_width - 10
        plot_y = 10
        
        # Background
        cv2.rectangle(frame, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height), (255, 255, 255), 1)
        
        # Draw grid lines
        cv2.line(frame, (plot_x, plot_y + plot_height // 2), (plot_x + plot_width, plot_y + plot_height // 2), (100, 100, 100), 1)
        
        # Draw FPS values as line graph
        if len(self.fps_history) > 1:
            max_fps = 60  # Reference max FPS
            for i in range(len(self.fps_history) - 1):
                x1 = plot_x + int((i / len(self.fps_history)) * plot_width)
                y1 = plot_y + plot_height - int((self.fps_history[i] / max_fps) * plot_height)
                x2 = plot_x + int(((i + 1) / len(self.fps_history)) * plot_width)
                y2 = plot_y + plot_height - int((self.fps_history[i + 1] / max_fps) * plot_height)
                
                # Clamp y values
                y1 = np.clip(y1, plot_y, plot_y + plot_height)
                y2 = np.clip(y2, plot_y, plot_y + plot_height)
                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display current FPS and label
        cv2.putText(frame, f"FPS: {fps:.1f}", (plot_x + 5, plot_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def _display_track(self, track, frame):
        x1, y1, x2, y2 = map(int, track.to_tlbr())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Display track_id if available (from deep sort tracks)
        if hasattr(track, 'track_id'):
            track_id = track.track_id
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        else:
            # MOSSE tracker
            cv2.putText(
                frame,
                f"ID: {self.id_to_track} (MOSSE)",
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

        # Initialize MOSSE tracker with the selected target
        for track in self.tracker.tracks:
            if track.track_id == self.id_to_track:
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                self._init_mosse(frame1, (x1, y1, x2, y2))
                break

        frames_to_skip = 10  # Run YOLO every 10 frames, use MOSSE for others
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
            
            # Draw FPS plot on frame
            self._draw_fps_plot(frame, current_fps)

            # Update MOSSE tracker every frame
            mosse_bbox = self._update_mosse(frame)
            
            # Run YOLO detections periodically
            frame_count += 1
            if frame_count % frames_to_skip == 0:
                self.tracker.predict()
                detections = self._get_detections(frame)
                self.tracker.update(detections)
                frame_count = 0

                # Update MOSSE with new detection from YOLO
                target_exists = False
                for track in self.tracker.tracks:
                    if track.track_id != self.id_to_track:
                        continue
                    print(f"Updating MOSSE with new detection for track ID {self.id_to_track}")
                    target_exists = True
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                    self._init_mosse(frame, (x1, y1, x2, y2))
                    mosse_bbox = (x1, y1, x2, y2)
                    break
            else:
                self.tracker.predict()

            # Use MOSSE bbox for tracking and alignment
            target_exists = False
            if mosse_bbox is not None:
                target_exists = True
                mosse_track = SimpleBBox(mosse_bbox)
                self._display_track(mosse_track, frame)
                self._align_camera_to_track(mosse_track, frame.shape, dt)
                    
            if not target_exists:
                self.motors.move(0, 0)

            cv2.imshow("Rocket Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.motors.move(0, 0)
        self.motors.close()
        cv2.destroyAllWindows()