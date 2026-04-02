import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

import threading
import time

import common
from motor import SerialMotorController

FOV_X = np.deg2rad(50)
FOV_Y = np.deg2rad(34)

MOTOR_PORT = "/dev/tty.usbmodem11101"
MOTOR_BAUD = 115200
PAN_RATE_MAX = np.deg2rad(180)
TILT_RATE_MAX = np.deg2rad(180)
MOTOR_SMOOTHING = 0.25
TRACKING_GAIN = 5.0

class PID:
    def __init__(self, kp, ki, kd, integral_limit=None, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit

        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        return output

id_to_track = None
filtered_pan_freq = 0.0
filtered_tilt_freq = 0.0

def command_motors(motors, pan_rate, tilt_rate):
    global filtered_pan_freq, filtered_tilt_freq

    # Map angular rate (rad/s) to board frequency command.
    x_freq = (pan_rate / PAN_RATE_MAX) * common.MAX_FREQ
    y_freq = (tilt_rate / TILT_RATE_MAX) * common.MAX_FREQ

    if abs(x_freq) < 1.0:
        x_freq = 0.0
    if abs(y_freq) < 1.0:
        y_freq = 0.0

    filtered_pan_freq = MOTOR_SMOOTHING * x_freq + (1.0 - MOTOR_SMOOTHING) * filtered_pan_freq
    filtered_tilt_freq = MOTOR_SMOOTHING * y_freq + (1.0 - MOTOR_SMOOTHING) * filtered_tilt_freq

    if abs(filtered_pan_freq) < 0.5:
        filtered_pan_freq = 0.0
    if abs(filtered_tilt_freq) < 0.5:
        filtered_tilt_freq = 0.0

    motors.move(filtered_pan_freq, filtered_tilt_freq)

def prompt_for_tracking_id():
    global id_to_track
    id_to_track = int(input("Enter the tracking ID: "))

def get_detections(model, encoder, frame):
    bboxes = []
    scores = []
    # Classes = [0] for person only
    results = model(frame, conf=0.25)[0]
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
    
    features = encoder(frame, bboxes)

    detections = [
        Detection(bbox, score, feature)
        for bbox, score, feature in zip(bboxes, scores, features)
    ]

    return detections


def get_tracked_target(tracker, tracking_id):
    for track in tracker.tracks:
        if track.track_id != tracking_id:
            continue
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        return track

    return None

def main():
    global id_to_track
    parser = argparse.ArgumentParser(description="YOLOv11 + DeepSORT target tracking with motor control")
    parser.add_argument("--camera-index", type=int, default=1, help="Camera index to open")
    parser.add_argument("--motor-port", type=str, default="/dev/tty.usbmodem11101", help="Serial port for the motor controller")
    parser.add_argument("--motor-baud", type=int, default=115200, help="Baud rate for the motor controller")
    parser.add_argument("--motor-gain", type=float, default=1, help="Scale factor from image error to motor frequency")
    parser.add_argument("--invert-x", action="store_true", help="Invert X motor direction")
    parser.add_argument("--invert-y", action="store_true", help="Invert Y motor direction")
    args = parser.parse_args()

    # -------------------------------
    # Load YOLOv11 model
    # -------------------------------
    model = YOLO("yolo11n.pt")

    # -------------------------------
    # Initialize DeepSORT
    # -------------------------------
    max_cosine_distance = 0.4
    nn_budget = None

    encoder = gdet.create_box_encoder(
        "mars-small128.pb",
        batch_size=1
    )

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )

    tracker = Tracker(metric)

    motors = None
    if args.motor_port:
        motors = SerialMotorController(args.motor_port, args.motor_baud)
        motors.run()
    
    pan_pid = PID(2.4, 0.08, 0.18, integral_limit=np.deg2rad(12), output_limit=np.deg2rad(120))
    tilt_pid = PID(2.4, 0.08, 0.18, integral_limit=np.deg2rad(12), output_limit=np.deg2rad(120))
    pan, tilt = 0.0, 0.0

    # -------------------------------
    # Video / webcam
    # -------------------------------
    cap = cv2.VideoCapture(args.camera_index)  # default is second camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    ret, frame1 = cap.read()
    if not ret:
        return
    
    detections = get_detections(model, encoder, frame1)

    if len(detections) == 0:
        return

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if track.time_since_update > 1:
            continue

        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_id = track.track_id
        # print(f"Track ID: {track_id}, BBox: ({x1}, {y1}, {x2}, {y2})")
        frame1 = cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)
        frame1 = cv2.putText(
            frame1,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    prompt_thread = threading.Thread(target=prompt_for_tracking_id, daemon=True)
    prompt_thread.start()

    cv2.imshow("YOLOv11 + DeepSORT", frame1)
    while prompt_thread.is_alive():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    prompt_thread.join()
    
    cv2.destroyAllWindows()

    print(f"Selected tracking ID: {id_to_track}")

    frames_to_skip = 1
    frame_count = 0

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_time
        dt = np.clip(dt, 1e-3, 0.1)
        last_time = now

        tracker.predict()
        frame_count += 1
        if frame_count % frames_to_skip == 0:
            detections = get_detections(model, encoder, frame)

            tracker.update(detections)
            frame_count = 0

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        # Draw tracks
        for track in tracker.tracks:
            if track.track_id != id_to_track:
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            track_id = track.track_id
            
            time_since_update = time.time() - last_time
            last_time = time.time()

            fps = 1.0 / time_since_update if time_since_update > 0 else float('inf')
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

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

        if motors is not None:
            target_track = get_tracked_target(tracker, id_to_track)
            if target_track is not None:
                x1, y1, x2, y2 = target_track.to_tlbr()
                target_x = (x1 + x2) / 2.0
                target_y = (y1 + y2) / 2.0

                err_x = (target_x - frame_center_x) / (frame_width / 2)
                err_y = (frame_center_y - target_y) / (frame_height / 2)
                err_x = np.clip(err_x, -1.0, 1.0) * TRACKING_GAIN
                err_y = np.clip(err_y, -1.0, 1.0) * TRACKING_GAIN * 2

                # Ignore tiny errors near image center to reduce jitter.
                deadzone = 0.05
                if abs(err_x) < deadzone:
                    err_x = 0.0
                if abs(err_y) < deadzone:
                    err_y = 0.0

                # -----------------------
                # Convert to angle error
                # -----------------------
                err_pan = err_x * (FOV_X / 2)
                err_tilt = err_y * (FOV_Y / 2)

                # -----------------------
                # PID control
                # -----------------------
                pan_rate = pan_pid.update(err_pan, dt)
                tilt_rate = tilt_pid.update(err_tilt, dt)
                pan += pan_rate * dt
                tilt += tilt_rate * dt
                command_motors(motors, pan_rate, tilt_rate)
            else:
                motors.move(0, 0)

        cv2.imshow("YOLOv11 + DeepSORT", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if motors is not None:
        motors.move(0, 0)
        motors.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()