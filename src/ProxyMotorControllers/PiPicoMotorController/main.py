from machine import Pin, PWM
import time

# --- Setup two PWM outputs ---
pan_step_pwm = PWM(Pin(8))   # Pin for buzzer/motor 1
tilt_step_pwm = PWM(Pin(28))    # Pin for buzzer/motor 2

# (Optional enable) + direction pins
pan_enable = Pin(7, Pin.OUT)
tilt_enable = Pin(27, Pin.OUT)
pan_dir = Pin(9, Pin.OUT)
tilt_dir = Pin(26, Pin.OUT)

led = Pin(25, Pin.OUT)  # onboard LED for status indication

# Start with both off
pan_step_pwm.duty_u16(0)
tilt_step_pwm.duty_u16(0)
led.off() # turn off LED

def set_freq(pwm, freq):
    '''
    Sets the frequency of the given PWM object. If the frequency is close to zero, it turns off the PWM.
    '''
    if abs(freq) <= 10:        # off
        pwm.duty_u16(0)
        led.off()  # turn off LED
    else:
        pwm.freq(abs(freq))   # set frequency
        pwm.duty_u16(32768)   # 50% duty
        led.on()  # turn on LED

def update_motors(pan_freq, tilt_freq):
    # Update PWM
    set_freq(pan_step_pwm, pan_freq)
    set_freq(tilt_step_pwm, tilt_freq)

    # Update enable + direction pins
    pan_enable.value(1 if abs(pan_freq) <= 1 else 0)
    pan_dir.value(1 if pan_freq < 0 else 0)
    tilt_enable.value(1 if abs(tilt_freq) <= 1 else 0)
    tilt_dir.value(0 if tilt_freq < 0 else 1)

def stop_motors():
    update_motors(0, 0)
    led.off()  # turn off LED
    
def main():
    print("Starting board")
    led.on()
    time.sleep(0.1)
    led.off()

    while True:
        user_input = input().strip()  # example: "1000 2000"
        if not user_input or len(parts := user_input.split()) != 2:
            continue
        pan_freq = int(parts[0])
        tilt_freq = int(parts[1])
        update_motors(pan_freq, tilt_freq)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_motors()
        print("Exiting on user interrupt")
    except Exception as e:
        stop_motors()
        print("Error in main:", e)