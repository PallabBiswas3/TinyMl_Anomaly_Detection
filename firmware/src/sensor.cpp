#include "sensor.h"
#include <cmath>

// ── SensorManager::begin ──────────────────────────────────────────────────────

bool SensorManager::begin() {
    Serial.println("[Sensor] Initializing HC-SR04...");

    pinMode(HCSR04_TRIG_PIN, OUTPUT);
    pinMode(HCSR04_ECHO_PIN, INPUT);
    digitalWrite(HCSR04_TRIG_PIN, LOW);
    delayMicroseconds(2);

    // Test read
    RawReading r = read_hcsr04();
    if (r.valid) {
        Serial.printf("[Sensor] HC-SR04 OK — first reading: %.1f cm\n", r.distance_cm);
    } else {
        Serial.println("[Sensor] HC-SR04 WARNING: first reading out of range (check wiring)");
    }

    // INA219 (optional)
    // Wire.begin();
    // ina219_available_ = ina219.begin();
    ina219_available_ = false;
    Serial.println("[Sensor] INA219: not connected (power monitoring estimated)");

    return true;
}

// ── HC-SR04 single-shot distance read ────────────────────────────────────────
// Sends a 10μs pulse on TRIG, measures ECHO pulse duration
// Distance = (echo_us * 0.0343) / 2  (speed of sound ~343 m/s)

RawReading SensorManager::read_hcsr04() {
    RawReading r;
    r.timestamp_ms = millis();
    r.valid        = false;

    // Ensure TRIG is low for a clean pulse
    digitalWrite(HCSR04_TRIG_PIN, LOW);
    delayMicroseconds(2);

    // Send 10μs trigger pulse
    digitalWrite(HCSR04_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(HCSR04_TRIG_PIN, LOW);

    // Measure echo pulse duration
    uint32_t echo_us = pulseIn(HCSR04_ECHO_PIN, HIGH, HCSR04_TIMEOUT_US);

    if (echo_us == 0) {
        Serial.println("[Sensor] DEBUG: echo_us = 0 (TIMEOUT!)");
    } else {
        Serial.printf("[Sensor] DEBUG: echo_us = %lu\n", echo_us);
    }

    if (echo_us == 0) {
        // Timeout — object out of range
        r.distance_cm = -1.0f;
        return r;
    }

    r.distance_cm = echo_us * 0.01715f; // (echo_us * 343m/s) / 2 / 10000
    r.valid       = (r.distance_cm >= 2.0f && r.distance_cm <= 400.0f);
    return r;
}

// ── Window statistics ─────────────────────────────────────────────────────────

int SensorManager::window_count() const {
    return window_full_ ? WINDOW_SIZE : (int)window_idx_;
}

float SensorManager::window_mean() const {
    int n = window_count();
    if (n == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += window_[i];
    return sum / n;
}

float SensorManager::window_variance() const {
    int n = window_count();
    if (n < 2) return 0.0f;
    float mean = window_mean();
    float sum  = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = window_[i] - mean;
        sum += diff * diff;
    }
    return sum / n;
}

float SensorManager::window_std() const {
    return sqrtf(window_variance());
}

// ── Feature extraction ────────────────────────────────────────────────────────

SensorFeatures SensorManager::extract_features(float dist_cm, uint32_t ts_ms) {
    SensorFeatures f;
    f.timestamp_ms = ts_ms;
    f.valid        = true;
    f.distance_cm  = dist_cm;

    // Push into circular window
    window_[window_idx_ % WINDOW_SIZE] = dist_cm;
    window_idx_++;
    if (window_idx_ >= WINDOW_SIZE) window_full_ = true;

    // Jitter = std deviation of window
    f.jitter   = window_std();
    f.variance = window_variance();

    // Rate of change = |delta_cm| / delta_seconds
    if (prev_distance_ >= 0.0f && ts_ms > prev_timestamp_) {
        float delta_cm  = fabsf(dist_cm - prev_distance_);
        float delta_sec = (ts_ms - prev_timestamp_) / 1000.0f;
        f.rate_of_change = delta_cm / delta_sec;
    } else {
        f.rate_of_change = 0.0f;
    }

    prev_distance_  = dist_cm;
    prev_timestamp_ = ts_ms;

    return f;
}

// ── Public read ───────────────────────────────────────────────────────────────

SensorFeatures SensorManager::read() {
    // ── SIMULATION MODE ──
    // We cycle through different scenarios every 10 seconds to test the AI
    static uint32_t start_ts = millis();
    uint32_t elapsed = (millis() - start_ts) / 1000;
    float dist = 30.0f;

    if (elapsed < 10) {
        // SCENARIO 1: Normal (30cm, stable)
        dist = 30.0f + ((rand() % 100) / 200.0f); // tiny noise
        if (elapsed == 0) Serial.println("\n>>> [SIMULATION] Scenario: NORMAL (Stable 30cm)");
    } 
    else if (elapsed < 20) {
        // SCENARIO 2: Anomaly (Object too close)
        dist = 6.0f + ((rand() % 100) / 50.0f);
        if (elapsed == 10) Serial.println("\n>>> [SIMULATION] ANOMALY: Object Too Close!");
    }
    else if (elapsed < 30) {
        // SCENARIO 3: Anomaly (High Jitter/Vibration)
        dist = 30.0f + ((rand() % 1000) / 50.0f); // wild swings
        if (elapsed == 20) Serial.println("\n>>> [SIMULATION] ANOMALY: High Jitter/Vibration!");
    }
    else {
        // Reset cycle
        start_ts = millis();
    }

    return extract_features(dist, millis());
}

// ── Power ─────────────────────────────────────────────────────────────────────

PowerReading SensorManager::read_power() {
    PowerReading p;
    p.available  = ina219_available_;
    p.voltage_mV = 3300.0f;
    p.current_mA = 120.0f; // typical ESP32 active draw
    p.power_mW   = p.voltage_mV * p.current_mA / 1000.0f;

    // Real INA219:
    // p.voltage_mV = ina219.getBusVoltage_V() * 1000.0f;
    // p.current_mA = ina219.getCurrent_mA();
    // p.power_mW   = ina219.getPower_mW();

    return p;
}

// ── Print helpers ─────────────────────────────────────────────────────────────

void SensorManager::print(const SensorFeatures& f) {
    Serial.printf(
        "[Sensor] t=%lums  dist=%.1fcm  jitter=%.3f  roc=%.3f  var=%.4f\n",
        f.timestamp_ms, f.distance_cm,
        f.jitter, f.rate_of_change, f.variance
    );
}

void SensorManager::print_power(const PowerReading& p) {
    Serial.printf("[Power]  %.0fmV  %.1fmA  %.1fmW%s\n",
        p.voltage_mV, p.current_mA, p.power_mW,
        p.available ? "" : " (estimated)");
}
