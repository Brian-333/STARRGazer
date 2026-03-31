from ultralytics import YOLO
import cv2
import time

class YOLOTracker:
    def __init__(self, model_path, downsample_factor=1):
        # Load the pre-trained YOLOv11 model
        self.model = YOLO(model_path)
        self.downsample_factor = downsample_factor

    def predict(self, video_source=0, device='cuda:0') -> list:

        # Open the webcam (0 is usually the default camera)
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("Frame rate: ", cap.get(cv2.CAP_PROP_FPS))

        last_time = time.time()
        while cap.isOpened():
            print("Frame rate: ", cap.get(cv2.CAP_PROP_FPS), end='\r')
            success, frame = cap.read()
            frame = cv2.resize(frame, (frame.shape[1] // self.downsample_factor, frame.shape[0] // self.downsample_factor))
            if not success:
                break
            
            # Run YOLOv11 inference on the frame
            results = self.model.predict(frame, device=device, stream=True, show_conf=True, verbose=False)

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
            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time

            annotated_frame = cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv11 Webcam Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print()

        return results

if __name__ == "__main__":
    from pathlib import Path
    SRC_DIR = Path(__file__).parent
    MODEL_PATH = SRC_DIR / 'best.pt'
    tracker = YOLOTracker(MODEL_PATH, downsample_factor=2)
    tracker.track()