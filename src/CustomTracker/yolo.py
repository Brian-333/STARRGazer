from ultralytics import YOLO
import cv2
import time

class YOLOTracker:
    def __init__(self, model_path):
        # Load the pre-trained YOLOv11 model
        self.model = YOLO(model_path)

    def predict(self, video_source=0, device='cuda:0') -> list:

        # Open the webcam (0 is usually the default camera)
        cap = cv2.VideoCapture(video_source)

        last_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv11 inference on the frame
            results = self.model.predict(frame, device=device, stream=True, show_conf=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            time_diff = time.time() - last_time
            last_time = time.time()

            annotated_frame = cv2.putText(annotated_frame, f'FPS: {1/time_diff:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv11 Webcam Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        return results

if __name__ == "__main__":
    from pathlib import Path
    SRC_DIR = Path(__file__).parent
    MODEL_PATH = SRC_DIR / 'best.pt'
    tracker = YOLOTracker(MODEL_PATH)
    tracker.track()