#include <Arduino.h>

#define LED_PIN 2

// This variable lives in RTC memory and survives a software reset
RTC_DATA_ATTR int restartCount = 0; 
int blinkCount = 0;

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    
    // Track how many times we have booted
    restartCount++; 

    Serial.printf("\n>>> [SYSTEM] ESP32 Started (Boot #%d)\n", restartCount);
}

void loop() {
    
    ESP.restart(); 
    

    // If restartCount is 2 or more, it will just continue to blink here forever
}
