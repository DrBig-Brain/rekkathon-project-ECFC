#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <Adafruit_MPU6050.h>
#include <arduinoFFT.h>

// ================= USER CONFIGURATION =================
const char* ssid = "LAMEIMPALA8678";
const char* password = "55pG#260";
const char* serverUrl = "http://172.18.59.141:5000/log"; 

// CHANGE THIS LABEL when recording different states!
const char* dataLabel = "motor_bad"; 
// ======================================================

// DSP Configuration
const uint16_t samples = 128; 
const double samplingFrequency = 200; 
double vRealX[samples], vImagX[samples];
double vRealY[samples], vImagY[samples];
double vRealZ[samples], vImagZ[samples];

// Batching
const int BATCH_SIZE = 5; // Smaller batch for stability
int batchCount = 0;
String batchData = "[";

Adafruit_MPU6050 mpu;
ArduinoFFT<double> FFT_X = ArduinoFFT<double>(vRealX, vImagX, samples, samplingFrequency);
ArduinoFFT<double> FFT_Y = ArduinoFFT<double>(vRealY, vImagY, samples, samplingFrequency);
ArduinoFFT<double> FFT_Z = ArduinoFFT<double>(vRealZ, vImagZ, samples, samplingFrequency);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n✓ WiFi Connected");

  if (!mpu.begin()) { Serial.println("✗ MPU Fail"); while(1) delay(10); }
  
  // High sensitivity settings for vibration
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
}

void loop() {
  // 1. SAMPLE
  for (int i = 0; i < samples; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    vRealX[i] = a.acceleration.x;
    vRealY[i] = a.acceleration.y;
    vRealZ[i] = a.acceleration.z;
    vImagX[i] = 0; vImagY[i] = 0; vImagZ[i] = 0;
    delayMicroseconds(4800); // Approx 200Hz
  }

  // 2. PROCESS (FFT + Stats)
  processFFT(FFT_X);
  processFFT(FFT_Y);
  processFFT(FFT_Z);
  
  float rmsX = calcRMS(vRealX);
  float rmsY = calcRMS(vRealY);
  float rmsZ = calcRMS(vRealZ);
  float kurtZ = calcKurtosis(vRealZ);
  
  // 3. PACK (Create clear, useful labels)
  String json = "{";
  json += "\"label\":\"" + String(dataLabel) + "\",";
  json += "\"rms_x\":" + String(rmsX, 4) + ",";
  json += "\"rms_y\":" + String(rmsY, 4) + ",";
  json += "\"rms_z\":" + String(rmsZ, 4) + ",";
  json += "\"peak_freq_x\":" + String(FFT_X.majorPeak(), 2) + ",";
  json += "\"peak_freq_y\":" + String(FFT_Y.majorPeak(), 2) + ",";
  json += "\"peak_freq_z\":" + String(FFT_Z.majorPeak(), 2) + ",";
  json += "\"kurtosis_z\":" + String(kurtZ, 4);
  json += "}";

  // 4. BATCH
  if (batchCount > 0) batchData += ",";
  batchData += json;
  batchCount++;

  Serial.printf("Sampled: RMS_Z=%.2f | Peak_Z=%.2f Hz\n", rmsZ, FFT_Z.majorPeak());

  // 5. SEND
  if (batchCount >= BATCH_SIZE) {
    sendBatch();
    batchData = "[";
    batchCount = 0;
  }
}

// --- Helpers ---
void processFFT(ArduinoFFT<double> &fft) {
  fft.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  fft.compute(FFT_FORWARD);
  fft.complexToMagnitude();
}

float calcRMS(double* data) {
  double sum = 0;
  for (int i=0; i<samples; i++) sum += data[i]*data[i];
  return sqrt(sum/samples);
}

float calcKurtosis(double* data) {
  double mean = 0, var = 0, fourth = 0;
  for (int i=0; i<samples; i++) mean += data[i];
  mean /= samples;
  for (int i=0; i<samples; i++) {
    double diff = data[i] - mean;
    var += diff*diff;
    fourth += diff*diff*diff*diff;
  }
  var /= samples; fourth /= samples;
  if (var < 0.0001) return 0;
  return (fourth / (var*var)) - 3.0;
}

void sendBatch() {
  if(WiFi.status() != WL_CONNECTED) return;
  WiFiClient client;
  HTTPClient http;
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(batchData + "]");
  Serial.printf(">> Batch Sent. Code: %d\n", code);
  http.end();
}