#pragma once
// HC-SR04 acquisition (or a realistic simulator). Produces RAW readings only -
// all cleaning happens in SignalPreprocessor so it is identical to training.

#include <Arduino.h>

// 1 = simulated sensor (no hardware needed), 0 = real HC-SR04.
// Set from platformio.ini build_flags.
#ifndef SENSOR_SIMULATION
#define SENSOR_SIMULATION 0
#endif

// ── HC-SR04 pin config ────────────────────────────────────────────────────────
#define HCSR04_TRIG_PIN   5      // GPIO5 → TRIG
#define HCSR04_ECHO_PIN   18     // GPIO18 ← ECHO (through 1k/2k voltage divider)
#define HCSR04_TIMEOUT_US 25000  // ~4.3 m; also caps the blocking time of one read

struct RawReading {
    float    distance_cm;   // -1 = no echo (timeout); otherwise as measured, unvalidated
    uint32_t timestamp_ms;  // millis() at trigger time
};

struct PowerReading {
    float voltage_mV;
    float current_mA;
    float power_mW;
    bool  available;        // false → datasheet estimate, not a measurement
};

class SensorManager {
public:
    bool begin();
    RawReading read();                      // one reading (real or simulated)
    PowerReading read_power() const;

private:
    RawReading read_hcsr04();
    float      simulated_distance(uint32_t now_ms);
    uint8_t    sim_phase_ = 0xFF;
};
