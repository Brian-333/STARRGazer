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

        # New constants needed for tracking:
        self.prev_center = None
        self.prev_time = None
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.vel_alpha = 0.7
        self.known_width = 0.15

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
        # New: Now returning both rates to use in the velocity estimation
        return pan_rate, tilt_rate

    # ----------------------------------------------------------------
    # New-untested velocity estimation
    def _estimate_velocity(self, track, frame_shape, pan_rate, tilt_rate):
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        obj_x = (x1 + x2) / 2.0
        obj_y = (y1 + y2) / 2.0

        current_time = time.time()
        if self.prev_center is None:
            self.prev_center = (obj_x, obj_y)
            self.prev_time = current_time
            return {
                "vx_px": 0.0,
                "vy_px": 0.0,
                "speed_px": 0.0,
                "speed_m": 0.0,
                "pixel_to_meter": 0.0,
            }

        dt_vel = np.clip(current_time - self.prev_time, 1e-3, 1.0)
        dx = obj_x - self.prev_center[0]
        dy = obj_y - self.prev_center[1]

        vx = dx / dt_vel
        vy = dy / dt_vel

        frame_height, frame_width = frame_shape[:2]
        px_per_rad_x = frame_width / FOV_X
        px_per_rad_y = frame_height / FOV_Y

        cam_vx = pan_rate * px_per_rad_x
        cam_vy = -tilt_rate * px_per_rad_y

        vx_corrected = vx - cam_vx
        vy_corrected = vy - cam_vy

        self.vx_smooth = self.vel_alpha * self.vx_smooth + (1.0 - self.vel_alpha) * vx_corrected
        self.vy_smooth = self.vel_alpha * self.vy_smooth + (1.0 - self.vel_alpha) * vy_corrected

        speed_px = np.sqrt(self.vx_smooth**2 + self.vy_smooth**2)
        width = float(x2 - x1)
        pixel_to_meter = self.known_width / width if width > 0 else 0.0
        vx_m = self.vx_smooth * pixel_to_meter
        vy_m = self.vy_smooth * pixel_to_meter
        speed_m = np.sqrt(vx_m**2 + vy_m**2)

        self.prev_center = (obj_x, obj_y)
        self.prev_time = current_time

        return {
            "vx_px": self.vx_smooth,
            "vy_px": self.vy_smooth,
            "speed_px": speed_px,
            "speed_m": speed_m,
            "pixel_to_meter": pixel_to_meter,
        }
    # New display of velocity
    def _display_velocity(self, frame, velocity):
        cv2.putText(frame, f"vx: {velocity['vx_px']:.1f} px/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"vy: {velocity['vy_px']:.1f} px/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"speed: {velocity['speed_px']:.1f} px/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Speed: {velocity['speed_m']:.2f} m/s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"scale: {velocity['pixel_to_meter']:.4f} m/px", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            if frame_count % frames_to_skip == 0:
                detections = self._get_detections(frame)

                self.tracker.update(detections)
                frame_count = 0

            target_exists = False
            for track in self.tracker.tracks:
                if track.track_id != self.id_to_track:
                    continue
                
                target_exists = True
                self._display_track(track, frame)

                # New velocity calls
                pan_rate, tilt_rate = self._align_camera_to_track(track, frame.shape, dt)
                velocity = self._estimate_velocity(track, frame.shape, pan_rate, tilt_rate)
                self._display_velocity(frame, velocity)
                    
            if not target_exists:
                # reset velocity constants
                self.prev_center = None
                self.prev_time = None
                self.vx_smooth = 0.0
                self.vy_smooth = 0.0
                self.motors.move(0, 0)

            cv2.imshow("Rocket Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.motors.move(0, 0)
        self.motors.close()
        cv2.destroyAllWindows()