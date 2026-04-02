import cv2
import torch
import numpy as np
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

import threading

id_to_track = None

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

def main():
    global id_to_track
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
    cap = cv2.VideoCapture(1)  # 1 = second camera

    ret, frame1 = cap.read()
    if not ret:
        return
    
    detections = get_detections(model, encoder, frame1)

    if len(detections) == 0:
        return

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
    cv2.waitKey(0)
    
    prompt_thread.join()
    
    cv2.destroyAllWindows()

    print(f"Selected tracking ID: {id_to_track}")

    frames_to_skip = 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracker.predict()
        frame_count += 1
        if frame_count % frames_to_skip == 0:
            detections = get_detections(model, encoder, frame)

            tracker.update(detections)
            frame_count = 0

        # Draw tracks
        for track in tracker.tracks:
            if track.track_id != id_to_track:
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()