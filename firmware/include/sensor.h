#pragma once
#include <Arduino.h>

// ── HC-SR04 pin config ────────────────────────────────────────────────────────
#define HCSR04_TRIG_PIN   5    // GPIO5 → TRIG
#define HCSR04_ECHO_PIN   18   // GPIO18 ← ECHO
#define HCSR04_TIMEOUT_US 25000  // ~4m max range

// Window size for feature extraction
#define WINDOW_SIZE  8

// ── Sensor reading (raw) ──────────────────────────────────────────────────────
struct RawReading {
    float    distance_cm;   // measured distance
    uint32_t timestamp_ms;  // millis()
    bool     valid;         // false if out of range / timeout
};

// ── Feature vector (input to the model) ──────────────────────────────────────
struct SensorFeatures {
    float distance_cm;      // latest distance
    float jitter;           // std deviation over window
    float rate_of_change;   // |delta| per second vs previous reading
    float variance;         // variance over window
    uint32_t timestamp_ms;
    bool valid;
};

// ── Power reading (INA219 optional) ──────────────────────────────────────────
struct PowerReading {
    float voltage_mV;
    float current_mA;
    float power_mW;
    bool  available;
};

// ── Sensor Manager ────────────────────────────────────────────────────────────
class SensorManager {
public:
    bool begin();

    // Take one HC-SR04 reading and update internal window
    // Returns extracted feature vector (ready for model input)
    SensorFeatures read();

    PowerReading read_power();

    static void print(const SensorFeatures& f);
    static void print_power(const PowerReading& p);

private:
    // Circular window of raw distance readings
    float    window_[WINDOW_SIZE] = {};
    uint32_t window_idx_          = 0;
    bool     window_full_         = false;
    float    prev_distance_       = -1.0f;
    uint32_t prev_timestamp_      = 0;
    bool     ina219_available_    = false;

    // HC-SR04 single-shot read
    RawReading read_hcsr04();

    // Feature extraction from current window
    SensorFeatures extract_features(float distance_cm, uint32_t timestamp_ms);

    // Window statistics
    float window_std()      const;
    float window_variance() const;
    float window_mean()     const;
    int   window_count()    const;
};
