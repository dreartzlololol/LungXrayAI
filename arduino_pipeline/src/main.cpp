#include <Arduino.h>

#define PCLK_PIN 4
#define VSYNC_PIN 2   // ยังไม่ต่อก็ได้

unsigned long lastPrint = 0;

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("EXT CLOCK CHECK");
  pinMode(PCLK_PIN, INPUT);
  pinMode(VSYNC_PIN, INPUT);
}

void loop() {
  if (millis() - lastPrint > 500) {
    lastPrint = millis();
    Serial.print("PCLK=");
    Serial.print(digitalRead(PCLK_PIN));
    Serial.print(" VSYNC=");
    Serial.println(digitalRead(VSYNC_PIN));
  }
}
