#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <Adafruit_MPU6050.h>
#include <arduinoFFT.h>

// WiFi Configuration
const char* ssid = "LAMEIMPALA8678";
const char* password = "55pG#260";
const char* serverUrl = "http://172.18.59.141:5000/data"; 

// FFT Configuration
const uint16_t samples = 128; // Increased for better frequency resolution
double vRealX[samples], vImagX[samples];
double vRealY[samples], vImagY[samples];
double vRealZ[samples], vImagZ[samples];
const double samplingFrequency = 200; // Increased to capture higher frequencies

// Feature storage for batch sending
const int BATCH_SIZE = 10; // Send 10 readings at once
int batchCount = 0;
String batchData = "[";

Adafruit_MPU6050 mpu;
ArduinoFFT<double> FFT_X = ArduinoFFT<double>(vRealX, vImagX, samples, samplingFrequency);
ArduinoFFT<double> FFT_Y = ArduinoFFT<double>(vRealY, vImagY, samples, samplingFrequency);
ArduinoFFT<double> FFT_Z = ArduinoFFT<double>(vRealZ, vImagZ, samples, samplingFrequency);

// Operating modes
enum Mode { TRAINING, PREDICTION };
Mode currentMode = TRAINING; // Start in training mode

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║  Industrial Anomaly Detector v2.0     ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  // WiFi Connection
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi Connected!");
    Serial.println("  IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n✗ WiFi Failed! Running in offline mode.");
  }

  // MPU6050 Initialization
  if (!mpu.begin()) {
    Serial.println("✗ MPU6050 not found! Check wiring.");
    while (1) { delay(1000); }
  }
  
  // Configure sensor settings for better vibration detection
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("✓ MPU6050 Ready");
  Serial.println("  Range: ±8g");
  Serial.println("  Gyro: ±500°/s");
  Serial.println("  Filter: 21Hz\n");
  
  Serial.println("Commands:");
  Serial.println("  't' - Switch to TRAINING mode");
  Serial.println("  'p' - Switch to PREDICTION mode");
  Serial.println("════════════════════════════════════════\n");
}

void loop() {
  // Check for mode change commands
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 't' || cmd == 'T') {
      currentMode = TRAINING;
      Serial.println("\n► MODE: TRAINING");
    } else if (cmd == 'p' || cmd == 'P') {
      currentMode = PREDICTION;
      Serial.println("\n► MODE: PREDICTION");
    }
  }

  // Sample accelerometer data
  Serial.println("\n[1] Sampling vibration data...");
  unsigned long startTime = micros();
  
  for (int i = 0; i < samples; i++) {
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    
    // Store all three axes for comprehensive analysis
    vRealX[i] = accel.acceleration.x;
    vRealY[i] = accel.acceleration.y;
    vRealZ[i] = accel.acceleration.z;
    
    vImagX[i] = 0;
    vImagY[i] = 0;
    vImagZ[i] = 0;
    
    // Precise sampling interval
    delayMicroseconds(5000 - (micros() % 5000)); // ~200Hz sampling
  }
  
  unsigned long samplingTime = micros() - startTime;
  Serial.printf("  Sampling took: %.2f ms\n", samplingTime / 1000.0);

  // Compute FFT for all axes
  Serial.println("[2] Computing FFT (3-axis)...");
  
  // X-axis
  FFT_X.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT_X.compute(FFT_FORWARD);
  FFT_X.complexToMagnitude();
  double peakX = FFT_X.majorPeak();
  
  // Y-axis
  FFT_Y.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT_Y.compute(FFT_FORWARD);
  FFT_Y.complexToMagnitude();
  double peakY = FFT_Y.majorPeak();
  
  // Z-axis
  FFT_Z.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT_Z.compute(FFT_FORWARD);
  FFT_Z.complexToMagnitude();
  double peakZ = FFT_Z.majorPeak();

  // Extract features
  float rmsX = calculateRMS(vRealX, samples);
  float rmsY = calculateRMS(vRealY, samples);
  float rmsZ = calculateRMS(vRealZ, samples);
  
  float kurtosisZ = calculateKurtosis(vRealZ, samples);
  
  Serial.printf("  Peak Frequencies: X=%.2f Hz, Y=%.2f Hz, Z=%.2f Hz\n", peakX, peakY, peakZ);
  Serial.printf("  RMS Values: X=%.3f, Y=%.3f, Z=%.3f\n", rmsX, rmsY, rmsZ);
  Serial.printf("  Kurtosis (Z): %.3f\n", kurtosisZ);

  // Build JSON for current reading
  String currentReading = buildFeatureJSON(peakX, peakY, peakZ, rmsX, rmsY, rmsZ, kurtosisZ);
  
  // Add to batch
  if (batchCount > 0) batchData += ",";
  batchData += currentReading;
  batchCount++;

  // Send batch when ready
  if (batchCount >= BATCH_SIZE) {
    Serial.println("[3] Batch ready - sending to server...");
    sendBatchToServer();
    
    // Reset batch
    batchData = "[";
    batchCount = 0;
  } else {
    Serial.printf("  Batch progress: %d/%d\n", batchCount, BATCH_SIZE);
  }

  delay(1000); // Adjust based on your monitoring needs
}

String buildFeatureJSON(float px, float py, float pz, float rx, float ry, float rz, float kurt) {
  // Extract top 5 frequency magnitudes for each axis
  String spectrumX = "[", spectrumY = "[", spectrumZ = "[";
  
  for (int i = 1; i <= 5; i++) { // Skip DC component (index 0)
    spectrumX += String(vRealX[i], 4);
    spectrumY += String(vRealY[i], 4);
    spectrumZ += String(vRealZ[i], 4);
    if (i < 5) {
      spectrumX += ",";
      spectrumY += ",";
      spectrumZ += ",";
    }
  }
  spectrumX += "]";
  spectrumY += "]";
  spectrumZ += "]";

  String json = "{";
  json += "\"mode\":\"" + String(currentMode == TRAINING ? "training" : "prediction") + "\",";
  json += "\"timestamp\":" + String(millis()) + ",";
  json += "\"peak_freq_x\":" + String(px, 2) + ",";
  json += "\"peak_freq_y\":" + String(py, 2) + ",";
  json += "\"peak_freq_z\":" + String(pz, 2) + ",";
  json += "\"rms_x\":" + String(rx, 4) + ",";
  json += "\"rms_y\":" + String(ry, 4) + ",";
  json += "\"rms_z\":" + String(rz, 4) + ",";
  json += "\"kurtosis\":" + String(kurt, 4) + ",";
  json += "\"spectrum_x\":" + spectrumX + ",";
  json += "\"spectrum_y\":" + spectrumY + ",";
  json += "\"spectrum_z\":" + spectrumZ;
  json += "}";
  
  return json;
}

void sendBatchToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("  ✗ WiFi disconnected - attempting reconnect...");
    WiFi.reconnect();
    delay(2000);
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("  ✗ Reconnect failed - data lost");
      return;
    }
  }

  WiFiClient client;
  HTTPClient http;
  
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000); // 10 second timeout

  String payload = batchData + "]";
  
  Serial.println("  Payload size: " + String(payload.length()) + " bytes");
  
  int httpCode = http.POST(payload);
  
  if (httpCode > 0) {
    Serial.printf("  ✓ Server response: %d\n", httpCode);
    
    if (httpCode == 200) {
      String response = http.getString();
      Serial.println("  Server says: " + response);
    }
  } else {
    Serial.printf("  ✗ POST failed: %s\n", http.errorToString(httpCode).c_str());
  }
  
  http.end();
}

// Statistical feature calculations
float calculateRMS(double* data, int length) {
  double sum = 0;
  for (int i = 0; i < length; i++) {
    sum += data[i] * data[i];
  }
  return sqrt(sum / length);
}

float calculateKurtosis(double* data, int length) {
  // Calculate mean
  double mean = 0;
  for (int i = 0; i < length; i++) {
    mean += data[i];
  }
  mean /= length;
  
  // Calculate variance and fourth moment
  double variance = 0;
  double fourthMoment = 0;
  
  for (int i = 0; i < length; i++) {
    double diff = data[i] - mean;
    variance += diff * diff;
    fourthMoment += diff * diff * diff * diff;
  }
  
  variance /= length;
  fourthMoment /= length;
  
  // Kurtosis = (fourth moment / variance^2) - 3
  if (variance < 0.0001) return 0; // Avoid division by zero
  return (fourthMoment / (variance * variance)) - 3.0;
}