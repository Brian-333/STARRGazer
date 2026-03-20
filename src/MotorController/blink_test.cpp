#include <Arduino.h>

void blink_setup() {
  // put your setup code here, to run once:
  Serial.begin (115200);
  Serial.println ("Blink with ESP32");

  pinMode (LED_BUILTIN, OUTPUT);
}

void blink_loop(int delay_time) {
  // put your main code here, to run repeatedly:
  digitalWrite (LED_BUILTIN, HIGH);
  delay (delay_time);
  digitalWrite (LED_BUILTIN, LOW);
  delay (delay_time);
}