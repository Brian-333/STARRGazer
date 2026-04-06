# --- HostMotorController.py ---
# Date: 2026-04-01
# Author: Brian Lin
# Description: This file contains the HostMotorController class which is used to control the pan and tilt
# This controller works only with the same motor controller code in the ESP32MotorController.ino file.
# It is not a general motor controller for any motor controller code.
import time
import serial
import subprocess
import threading

MAX_RETRIES = 5


class HostMotorController:
    def __init__(self, port, baudrate=115200, pan_step_pin=12, pan_dir_pin=14, tilt_step_pin=16, tilt_dir_pin=18, steps_per_revolution=200):
        self.port = port
        self.baudrate = baudrate
        self.pan_step_pin = pan_step_pin
        self.pan_dir_pin = pan_dir_pin
        self.tilt_step_pin = tilt_step_pin
        self.tilt_dir_pin = tilt_dir_pin
        self.steps_per_revolution = steps_per_revolution

        self.serial_connection = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)

    def setup(self):
        # Set pin modes for motor control
        self.send_command(f"SET_PIN_MODE {self.pan_step_pin} OUTPUT")
        self.send_command(f"SET_PIN_MODE {self.pan_dir_pin} OUTPUT")
        self.send_command(f"SET_PIN_MODE {self.tilt_step_pin} OUTPUT")
        self.send_command(f"SET_PIN_MODE {self.tilt_dir_pin} OUTPUT")

    def move_pan(self, steps, direction):
        # Code to move the pan motor a certain number of steps in a given direction
        pass

    def move_tilt(self, steps, direction):
        # Code to move the tilt motor a certain number of steps in a given direction
        pass