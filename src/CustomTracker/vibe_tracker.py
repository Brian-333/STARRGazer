import cv2
import numpy as np
import time

# -----------------------
# PID Controller
# -----------------------
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


# -----------------------
# Camera setup
# -----------------------
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cx, cy = width // 2, height // 2

# Field of view (approximate)
FOV_X = np.deg2rad(60)
FOV_Y = np.deg2rad(45)

# -----------------------
# PID for pan & tilt
# -----------------------
pan_pid = PID(4.0, 0.2, 0.35, integral_limit=np.deg2rad(20), output_limit=np.deg2rad(180))
tilt_pid = PID(4.0, 0.2, 0.35, integral_limit=np.deg2rad(20), output_limit=np.deg2rad(180))

pan, tilt = 0.0, 0.0

last_time = time.time()

# -----------------------
# Tracker (CSRT)
# -----------------------
tracker = cv2.legacy.TrackerKCF_create()
initBB = None

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - last_time
    dt = np.clip(dt, 1e-3, 0.1)
    last_time = now

    # Select object manually
    if initBB is None:
        cv2.putText(frame, "Press 's' to select object", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    else:
        success, box = tracker.update(frame)

        if success:
            x, y, w, h = map(int, box)
            obj_x = x + w // 2
            obj_y = y + h // 2

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # -----------------------
            # ERROR (pixels → normalized)
            # -----------------------
            err_x = (obj_x - cx) / (width / 2)
            err_y = (cy - obj_y) / (height / 2)
            err_x = np.clip(err_x, -1.0, 1.0)
            err_y = np.clip(err_y, -1.0, 1.0)

            # Ignore tiny errors near image center to reduce jitter.
            deadzone = 0.03
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

            # Clamp pan/tilt ranges to realistic gimbal limits.
            pan = np.clip(pan, np.deg2rad(-180), np.deg2rad(180))
            tilt = np.clip(tilt, np.deg2rad(-80), np.deg2rad(80))

            # -----------------------
            # Display info
            # -----------------------
            cv2.putText(frame, f"Pan: {np.rad2deg(pan):.1f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"Tilt: {np.rad2deg(tilt):.1f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Crosshair
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
        else:
            cv2.putText(frame, "Tracking lost", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Tracking", frame, fromCenter=False)
        tracker.init(frame, initBB)
        pan_pid.reset()
        tilt_pid.reset()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()