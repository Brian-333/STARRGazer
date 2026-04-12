import argparse
import math
import time

import pyglet

import HostMotorController.common as common
from HostMotorController.motor import SerialMotorController


DEFAULT_PORT = "/dev/tty.usbmodem11101"
DEFAULT_BAUD = 115200


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0

    sign = 1.0 if value >= 0.0 else -1.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return sign * max(0.0, min(1.0, scaled))


def shape_axis(value: float, deadzone: float, expo: float) -> float:
    value = max(-1.0, min(1.0, value))
    value = apply_deadzone(value, deadzone)
    if value == 0.0:
        return 0.0
    return math.copysign(abs(value) ** expo, value)


def find_logitech_joystick():
    devices = pyglet.input.get_joysticks()
    if not devices:
        return None

    for joy in devices:
        name = (joy.device.name or "").lower()
        if "logitech" in name and "extreme" in name:
            return joy

    return devices[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Manual gimbal control with Logitech Extreme 3D Pro using pyglet")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port for the motor controller")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.60,
        help="Global max output scale in [0, 1] applied to common.MAX_FREQ",
    )
    parser.add_argument("--deadzone", type=float, default=0.12, help="Axis deadzone in [0, 0.5)")
    parser.add_argument("--expo", type=float, default=1.6, help="Axis response curve exponent")
    parser.add_argument("--invert-y", action="store_true", help="Invert joystick Y axis")
    return parser.parse_args()


def main():
    args = parse_args()

    motor_controller = SerialMotorController(args.port, args.baud)
    motor_controller.run()

    joystick = find_logitech_joystick()
    if joystick is None:
        motor_controller.move(0, 0)
        time.sleep(0.3)
        motor_controller.close()
        raise RuntimeError("No joystick found. Connect Logitech Extreme 3D Pro and try again.")

    joystick.open()
    window = pyglet.window.Window(width=560, height=180, caption="STARRGazer Manual Control", resizable=False)
    output_label = pyglet.text.Label("", x=16, y=90, anchor_x="left", anchor_y="center")
    help_label = pyglet.text.Label(
        "Use stick twist for pan and Y for tilt. Trigger or ESC to stop and exit.",
        x=16,
        y=28,
        anchor_x="left",
        anchor_y="center",
    )

    state = {"pan": 0.0, "y": 0.0, "trigger": False}
    max_freq = common.MAX_FREQ * max(0.0, min(1.0, args.scale))
    deadzone = max(0.0, min(0.49, args.deadzone))
    expo = max(1.0, args.expo)

    @joystick.event
    def on_joyaxis_motion(_joystick, axis, value):
        # Logitech Extreme 3D Pro typically reports twist as "z" or "rz".
        if axis in ("z", "rz"):
            state["pan"] = value
        elif axis == "x" and state["pan"] == 0.0:
            # Fallback when twist axis is not exposed by the driver.
            state["pan"] = value
        elif axis == "y":
            state["y"] = value

    @joystick.event
    def on_joybutton_press(_joystick, button):
        if button == 0:
            state["trigger"] = True

    @window.event
    def on_key_press(symbol, _modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    @window.event
    def on_draw():
        window.clear()
        output_label.draw()
        help_label.draw()

    def update(_dt):
        pan = shape_axis(state["pan"], deadzone, expo)
        y = shape_axis(state["y"], deadzone, expo)

        if args.invert_y:
            y = -y

        x_freq = pan * max_freq
        y_freq = -y * max_freq

        motor_controller.move(x_freq, y_freq)
        output_label.text = (
            f"Joystick pan={state['pan']:+.2f}, y={state['y']:+.2f} | "
            f"Cmd x={x_freq:+.0f}, y={y_freq:+.0f}"
        )

        if state["trigger"]:
            pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 1.0 / 60.0)

    print(f"Using joystick: {joystick.device.name}")
    print("Press trigger or ESC to quit.")

    try:
        pyglet.app.run()
    finally:
        motor_controller.move(0, 0)
        time.sleep(0.3)
        motor_controller.close()


if __name__ == "__main__":
    main()