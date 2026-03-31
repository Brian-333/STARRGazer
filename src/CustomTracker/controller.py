import serial
import threading
import time

class SerialController:
    def __init__(self, port='COM6', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.thread = None

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for connection to establish
            print(f"Connected to {self.ser.port} at {self.ser.baudrate} baud.")
            self.thread = threading.Thread(target=self.__read_from_port, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"Failed to connect: {e}")

    def __read_from_port(self):
        while True:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line:
                    current_timestamp = time.time()
                    print(f"[{current_timestamp}][Device]: {line}")
            except Exception as e:
                print(f"Error reading: {e}")
                break

    def send_message(self, message):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((message + '\n').encode('utf-8'))
            except Exception as e:
                print(f"Error sending message: {e}")
        else:
            print("Serial port is not open.")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

        if self.thread and self.thread.is_alive():
            self.thread.join()

if __name__ == "__main__":
    controller = SerialController()
    controller.connect()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.close()