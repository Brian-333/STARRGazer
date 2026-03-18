import CustomTracker.yolo
from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SRC_DIR / 'best.pt'

def main():
    new_tracker = CustomTracker.yolo.YOLOTracker(MODEL_PATH)
    result = new_tracker.predict()


if __name__ == "__main__":
    main()