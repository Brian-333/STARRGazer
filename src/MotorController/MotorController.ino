#include <blink_test.h>

#define STEP1 25
#define DIR1  26

#define STEP2 32
#define DIR2  33

// Motor state
long current1 = 0, target1 = 0;
long current2 = 0, target2 = 0;

unsigned long lastStep1 = 0;
unsigned long lastStep2 = 0;

int stepDelay = 300; // speed (lower = faster)

// Settings
const int stepsPerRev = 3200; // adjust to your microstepping
const float degToSteps = stepsPerRev / 360.0;

// Serial buffer
String input = "";

void setup() {
  Serial.begin(115200);

  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);

  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);
  blink_setup();
}

void stepMotor(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(2);
  digitalWrite(stepPin, LOW);
}

// Convert angle → steps
long angleToSteps(float angle) {
  return angle * degToSteps;
}

// Parse "pan,tilt"
void handleSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      float pan, tilt;
      sscanf(input.c_str(), "%f,%f", &pan, &tilt);

      target1 = angleToSteps(pan);
      target2 = angleToSteps(tilt);

      input = "";
    } else {
      input += c;
    }
  }
}

void updateMotor1() {
  if (current1 == target1) return;

  if (micros() - lastStep1 >= stepDelay) {
    lastStep1 = micros();

    digitalWrite(DIR1, target1 > current1 ? HIGH : LOW);
    stepMotor(STEP1);

    current1 += (target1 > current1) ? 1 : -1;
  }
}

void updateMotor2() {
  if (current2 == target2) return;

  if (micros() - lastStep2 >= stepDelay) {
    lastStep2 = micros();

    digitalWrite(DIR2, target2 > current2 ? HIGH : LOW);
    stepMotor(STEP2);

    current2 += (target2 > current2) ? 1 : -1;
  }
}

unsigned long lastSend = 0;

void sendFeedback() {
  Serial.print(current1);
  Serial.print(",");
  Serial.print(current2);
  Serial.print(",");
  Serial.print(target1);
  Serial.print(",");
  Serial.println(target2);
}

void loop() {
  blink_loop(500);
  handleSerial();
  updateMotor1();
  updateMotor2();

  // Send feedback at ~20 Hz
  if (millis() - lastSend > 50) {
    lastSend = millis();
    sendFeedback();
  }
}