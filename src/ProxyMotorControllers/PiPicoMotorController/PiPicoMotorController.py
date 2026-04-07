from machine import Pin, PWM
import select
import sys
import time

# --- Setup two PWM outputs ---
pan_step_pwm = PWM(Pin(7))   # Pin for buzzer/motor 1
tilt_step_pwm = PWM(Pin(28))    # Pin for buzzer/motor 2

# Optional enable + direction pins
pan_enable = Pin(8, Pin.OUT)
tilt_enable = Pin(27, Pin.OUT)
pan_dir = Pin(9, Pin.OUT)
tilt_dir = Pin(26, Pin.OUT)

led = Pin("LED", Pin.OUT)  # onboard LED for status indication

# Start with both off
pan_step_pwm.duty_u16(0)
tilt_step_pwm.duty_u16(0)
led.off() # turn off LED

CONNECTION_TIMEOUT = 1  # seconds to wait for input before stopping motors
motors_running = False

# Set up polling for stdin
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)

def set_freq(pwm, freq):
    '''
    Sets the frequency of the given PWM object. If the frequency is close to zero, it turns off the PWM.
    '''
    global motors_running
    if abs(freq) <= 10:        # off
        pwm.duty_u16(0)
        motors_running = False
    else:
        pwm.freq(abs(freq))   # set frequency
        pwm.duty_u16(32768)   # 50% duty
        motors_running = True
        led.on()  # turn on LED

def update_motors(pan_freq, tilt_freq):
    # Update PWM
    set_freq(pan_step_pwm, pan_freq)
    set_freq(tilt_step_pwm, tilt_freq)

    # Update enable + direction pins
    pan_enable.value(1 if abs(pan_freq) <= 1 else 0)
    pan_dir.value(0 if pan_freq < 0 else 1) # different dir for x axis
    tilt_enable.value(1 if abs(tilt_freq) <= 1 else 0)
    tilt_dir.value(0 if tilt_freq < 0 else 1)

def stop_motors():
    update_motors(pan_step_pwm.freq() // 2, tilt_step_pwm.freq() // 2)  # this will set duty to 0 if freq is low
    time.sleep(0.1)  # small delay to ensure motors stop
    update_motors(0, 0)
    led.off()  # turn off LED

def safe_input(timeout=CONNECTION_TIMEOUT):
    '''
    Waits for user input with a timeout. If no input is received within the timeout, it raises a TimeoutError.
    '''

    events = poller.poll(timeout * 1000)  # timeout in milliseconds
    if events:
        input_str = sys.stdin.readline().strip()
        return input_str
    else:
        return None  # indicate timeout
    
def parse_input(input_str):
    parts = input_str.split()
    mode = parts[0].strip().lower()
    if mode == "move":
        if len(parts) != 3:
            print("Invalid input format. Expected: 'move pan_freq tilt_freq'")
            return
        pan_freq = int(parts[1].strip())
        tilt_freq = int(parts[2].strip())
        update_motors(pan_freq, tilt_freq)
    elif mode == "stop":
        stop_motors()
    elif mode == "timeout":
        global CONNECTION_TIMEOUT
        CONNECTION_TIMEOUT = int(parts[1].strip())
        print(f"Connection timeout set to {CONNECTION_TIMEOUT} seconds")
    else:
        print("Unknown command. Use 'move', 'stop', or 'timeout'.")

    return

def main():
    global motors_running
    print("Starting board")
    start_time = time.ticks_ms()
    while (time.ticks_ms() - start_time) < 3000:  # wait up to 5 seconds for board to initialize
        led.toggle()
        time.sleep(0.5)

    while True:
        user_input = safe_input(timeout=CONNECTION_TIMEOUT)  # example: "1000 2000"
        if not user_input and motors_running:
            print("No input received, stopping motors for safety")
            stop_motors()
            continue

        if user_input:
            parse_input(user_input)

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        stop_motors()
        print("Exiting on user interrupt")
    except Exception as e:
        stop_motors()
        print("Error in main:", e)