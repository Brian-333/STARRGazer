from pathlib import Path
from RocketTracker import RocketTracker
from ultralytics import YOLO
from deep_sort.tools import generate_detections as gdet

SRC_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SRC_DIR / 'yolo26n.pt'
ENCODER_MODEL_PATH = SRC_DIR / 'mars-small128.pb'

MOTOR_PORT = "/dev/tty.usbmodem11101"
MOTOR_BAUD = 115200

def main():
    yolo = YOLO(MODEL_PATH)
    encoder = gdet.create_box_encoder(ENCODER_MODEL_PATH, batch_size=1)
    tracker = RocketTracker(yolo, encoder, MOTOR_PORT, MOTOR_BAUD)
    tracker.run(0)

if __name__ == "__main__":
    main()