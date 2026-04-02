import cv2
import numpy as np
import time
import math

# This is a basic velocity estimation script with a static webcam for now
# This needs to be integrated with the tracker as it's own class and we need to account for the pan-tilt

cap = cv2.VideoCapture(0)

# Camera parameters
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cx, cy = width // 2, height // 2

# Constants for conversion (these would need to be calibrated for your specific setup)
KNOWN_WIDTH = 0.15  # meters (width of the object being tracked)
FOCAL_LENGTH = 700  # pixels (this would need to be calibrated)

prev_center = None
prev_time = time.time()

# Initialize tracker
tracker = cv2.TrackerCSRT_create() # Play around with other trackers

# Read the first frame - select bounding box
ret, frame = cap.read()
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

# Loop through the frames
while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)
    current_time = time.time()

    if success:
        # Velocity estimation steps:
        # 3. Get centroid and calculate distance
        x, y, w, h = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)
        
        # 4. Convert pixels to meters
        # Ratio = KNOWN_WIDTH / Pixel width of bounding box
        pixel_to_meter_ratio = KNOWN_WIDTH / w

        if prev_center is not None:
            # Calculate distance moved in pixels
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            pixel_distance = math.sqrt(dx**2 + dy**2)

            # Convert to meters
            real_distance = pixel_distance * pixel_to_meter_ratio

            # Calculate time elapsed
            time_elapsed = current_time - prev_time

            # Avoid division by zero
            if time_elapsed > 0:
                 # 5. Calculate velcoity: speed = dist / time
                speed = real_distance / time_elapsed
            
            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Display the FPS of the camera and the tracker
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Update previous states
        prev_center = center
        prev_time = current_time

        # Draw bounding box and centroid
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('Tracking Frame', frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


       
        

