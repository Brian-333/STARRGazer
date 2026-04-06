// ===== ESP32 PAN-TILT TB6600 CONTROL =====
#include <Arduino.h>

#define PAN_STEP 18
#define PAN_DIR  19

#define TILT_STEP 23
#define TILT_DIR  22

#define ENABLE_PIN 21

#define LED_PIN 2   // built-in LED (change if needed)

short current_mode = 0;

unsigned long lastSerialTime = 0;
unsigned long timeout = 1000; // 1 second timeout

bool connectionAlive = false;

void setup() {
  Serial.begin(115200);

  pinMode(PAN_STEP, OUTPUT);
  pinMode(PAN_DIR, OUTPUT);
  pinMode(TILT_STEP, OUTPUT);
  pinMode(TILT_DIR, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(ENABLE_PIN, HIGH); // enable driver

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
}

void initialization() {

}

void stepMotor(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(400);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(400);
}

void stopMotors() {
  // Stop movement immediately
  target_pan = pan_position;
  target_tilt = tilt_position;

  // Disable driver (no holding torque)
  digitalWrite(ENABLE_PIN, LOW);
}

void moveAxis(long &pos, long target, int stepPin, int dirPin) {
  if (pos == target) return;

  if (target > pos) {
    digitalWrite(dirPin, HIGH);
    pos++;
  } else {
    digitalWrite(dirPin, LOW);
    pos--;
  }

  stepMotor(stepPin);
}

void loop() {
  // Receive: "pan,tilt\n"
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    // mark connection active
    lastSerialTime = millis();
    connectionAlive = true;

    // LED ON
    digitalWrite(LED_PIN, HIGH);

    sscanf(data.c_str(), "%ld,%ld", &target_pan, &target_tilt);
  }


  // ===== CONNECTION LOST =====
  if (millis() - lastSerialTime > timeout) {

    if (connectionActive) {
      // Only trigger once
      stopMotors();
      connectionActive = false;
    }
    digitalWrite(LED_PIN, LOW);
  }

  // ===== MOVE ONLY IF CONNECTED =====
  if (connectionActive) {
    moveAxis(pan_position, target_pan, PAN_STEP, PAN_DIR);
    moveAxis(tilt_position, target_tilt, TILT_STEP, TILT_DIR);
  }
}