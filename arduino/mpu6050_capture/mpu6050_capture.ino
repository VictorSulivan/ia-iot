/*
 * MPU6050 gesture capture firmware
 *
 * Wiring (ESP32 / boards with flexible Wire pins):
 *   SDA -> pin 8
 *   SCL -> pin 9
 *
 * Protocol (115200 baud):
 *   Host sends 's'
 *   Device streams NUM_SAMPLES lines: ax,ay,az,gx,gy,gz
 *   Device ends with: DONE
 *
 * Library: Electronic Cats MPU6050 (or equivalent MPU6050.h)
 */

#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

const int SDA_PIN = 8;
const int SCL_PIN = 9;
const int NUM_SAMPLES = 200;
const int INTERVAL_MS = 30; // ~6 s per capture

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  mpu.initialize();
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 's') {
      for (int i = 0; i < NUM_SAMPLES; i++) {
        int16_t ax, ay, az;
        int16_t gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        Serial.print(ax); Serial.print(",");
        Serial.print(ay); Serial.print(",");
        Serial.print(az); Serial.print(",");
        Serial.print(gx); Serial.print(",");
        Serial.print(gy); Serial.print(",");
        Serial.println(gz);
        delay(INTERVAL_MS);
      }
      Serial.println("DONE");
    }
  }
}
