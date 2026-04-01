import cv2
import torch
import numpy as np
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet


def main():
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

    # -------------------------------
    # Video / webcam
    # -------------------------------
    cap = cv2.VideoCapture(0)  # 0 = webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------
        # YOLO inference
        # -------------------------------
        results = model(frame, conf=0.25)[0]

        bboxes = []
        scores = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                w = x2 - x1
                h = y2 - y1

                bboxes.append([x1, y1, w, h])
                scores.append(conf)

        # -------------------------------
        # DeepSORT
        # -------------------------------
        if len(bboxes) > 0:
            features = encoder(frame, bboxes)

            detections = [
                Detection(bbox, score, feature)
                for bbox, score, feature in zip(bboxes, scores, features)
            ]

            tracker.predict()
            tracker.update(detections)

            # Draw tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

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

        cv2.imshow("YOLOv11 + DeepSORT", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()