import CustomTracker.yolo
from pathlib import Path
import argparse

SRC_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SRC_DIR / 'best.pt'

def main():
    args = argparse.ArgumentParser(description="Rocket Tracker")
    args.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (e.g., "cuda:0", "cpu" or "mps")')
    args.add_argument('--video_source', type=int, default=0, help='Video source (default is 0 for webcam)')
    args.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the YOLO model weights')
    args = args.parse_args()

    new_tracker = CustomTracker.yolo.YOLOTracker(args.model_path)
    result = new_tracker.predict(device=args.device)


if __name__ == "__main__":
    main()