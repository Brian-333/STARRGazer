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

        past_10_fps = []

        last_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv11 inference on the frame
            results = self.model.predict(frame, device=device, stream=True, show_conf=True)

            # Visualize the results on the frame
            annotated_frame = frame.copy()
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{self.model.names[int(cls_id)]}: {conf:.2f}'
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            time_diff = time.time() - last_time
            last_time = time.time()

            past_10_fps.append(1/time_diff)
            if len(past_10_fps) > 10:
                past_10_fps.pop(0)

            annotated_frame = cv2.putText(annotated_frame, f'FPS: {sum(past_10_fps)/len(past_10_fps):.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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