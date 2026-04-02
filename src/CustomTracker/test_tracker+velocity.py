import cv2
import numpy as np
import time

# Note for velocity tracking
'''
Measured velocity = object motion + camera motion
v_true = v_image - v_camera
'''


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
fps = 0.0
fps_smoothing = 0.9

last_time = time.time()

# -----------------------
# Tracker (CSRT)
# Try CSRT vs KCF
# -----------------------
tracker = cv2.TrackerKCF_create()
initBB = None

# -----------------------
# Velocity Integration
prev_center = None
prev_time = None

# Optional smoothing
vel_alpha = 0.7
vx_smooth, vy_smooth = 0.0, 0.0

# Width of the object(in meters)
KNOWN_WIDTH = 0.15
# -----------------------


# -----------------------
# Main loop
# -----------------------
cap.set(cv2.CAP_PROP_FPS, 30)
while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    if not ret:
        break

    now = time.time()
    dt = now - last_time
    dt = np.clip(dt, 1e-3, 0.1)
    last_time = now

    current_fps = 1.0 / dt
    fps = fps_smoothing * fps + (1 - fps_smoothing) * current_fps

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

            # ------------------------------
            # New-untested velocity tracking

            current_time = now  # reuse your timestamp

            if prev_center is not None:
                dt_vel = current_time - prev_time
                if dt_vel > 1e-3:
                    # -----------------------
                    # Raw pixel velocity
                    # -----------------------
                    dx = obj_x - prev_center[0]
                    dy = obj_y - prev_center[1]

                    vx = dx / dt_vel
                    vy = dy / dt_vel

                    # -----------------------
                    # Convert camera motion to pixel motion
                    # -----------------------
                    # radians → pixels
                    px_per_rad_x = width / FOV_X
                    px_per_rad_y = height / FOV_Y

                    cam_vx = pan_rate * px_per_rad_x
                    cam_vy = -tilt_rate * px_per_rad_y  # sign flip due to image coords

                    # -----------------------
                    # Compensated velocity
                    # -----------------------
                    vx_corrected = vx - cam_vx
                    vy_corrected = vy - cam_vy

                    # -----------------------
                    # Smooth velocity
                    # -----------------------
                    vx_smooth = vel_alpha * vx_smooth + (1 - vel_alpha) * vx_corrected
                    vy_smooth = vel_alpha * vy_smooth + (1 - vel_alpha) * vy_corrected


                    speed = np.sqrt(vx_smooth**2 + vy_smooth**2)

                    # -----------------------
                    # Approximate physical units (m/s) using known width
                    # (basic sanity check before stereo depth correction)
                    # -----------------------
                    if w > 0:
                        pixel_to_meter = KNOWN_WIDTH / float(w)  # m per pixel at current estimated distance
                        vx_m = vx_smooth * pixel_to_meter
                        vy_m = vy_smooth * pixel_to_meter
                        speed_m = np.sqrt(vx_m**2 + vy_m**2)
                    else:
                        pixel_to_meter = 0.0
                        vx_m = vy_m = speed_m = 0.0

                    # -----------------------
                    # Display
                    # -----------------------
                    cv2.putText(frame, f"vx: {vx_smooth:.1f} px/s", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"vy: {vy_smooth:.1f} px/s", (20, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"speed: {speed:.1f} px/s", (20, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(frame, f"Speed: {speed_m:.2f} m/s", (20, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"scale: {pixel_to_meter:.4f} m/px", (20, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Update history
            prev_center = (obj_x, obj_y)
            prev_time = current_time

            # ------------------------------

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # -----------------------
            # ERROR (pixels → normalized)
            # -----------------------
            '''
            err_x = (obj_x - cx) / (width / 2)
            err_y = (cy - obj_y) / (height / 2)
            err_x = np.clip(err_x, -1.0, 1.0)
            err_y = np.clip(err_y, -1.0, 1.0)
            '''

            # Addition --------------------------
            # Predictive tracking - This would be done by the Kalman filter in Deepsort
            # Goal is to reduce lag, stabalize the PID response, and improve the tracking of fast objects
            prediction_horizon = 0.1  # seconds ahead

            pred_x = obj_x + vx_smooth * prediction_horizon
            pred_y = obj_y + vy_smooth * prediction_horizon

            err_x = (pred_x - cx) / (width / 2)
            err_y = (cy - pred_y) / (height / 2)


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

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

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