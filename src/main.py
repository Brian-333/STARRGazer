from ultralytics import YOLO
import cv2
from pathlib import Path

SRC_DIR = Path(__file__).parent
MODEL_PATH = SRC_DIR / 'best.pt'

# 1. Load the pre-trained YOLOv11 model (nano version for speed)
model = YOLO('best.pt') 

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Run YOLOv11 inference on the frame
    results = model(frame)

    # 4. Visualize the results on the frame
    annotated_frame = results[0].plot()

    # 5. Display the annotated frame
    cv2.imshow("YOLOv11 Webcam Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()