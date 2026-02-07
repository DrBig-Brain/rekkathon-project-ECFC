#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <Adafruit_MPU6050.h>
#include <arduinoFFT.h>

// ================= USER CONFIGURATION =================
const char* ssid = "LAMEIMPALA8678";
const char* password = "55pG#260";
// Ensure this route matches your Python prediction endpoint
const char* serverUrl = "http://172.18.59.141:5000/predict"; 
// ======================================================

// DSP Configuration
const uint16_t samples = 128; 
const double samplingFrequency = 200; 
double vRealX[samples], vImagX[samples];
double vRealY[samples], vImagY[samples];
double vRealZ[samples], vImagZ[samples];

Adafruit_MPU6050 mpu;
ArduinoFFT<double> FFT = ArduinoFFT<double>(vRealX, vImagX, samples, samplingFrequency); 
// Note: Created a single FFT object to reuse (memory optimization)

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n✓ WiFi Connected");

  if (!mpu.begin()) { Serial.println("✗ MPU Fail"); while(1) delay(10); }
  
  // Settings
  mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
}

void loop() {
  // 1. SAMPLE (~0.6 seconds)
  for (int i = 0; i < samples; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    vRealX[i] = a.acceleration.x;
    vRealY[i] = a.acceleration.y;
    vRealZ[i] = a.acceleration.z;
    // Clear imaginary parts
    vImagX[i] = 0; vImagY[i] = 0; vImagZ[i] = 0;
    delayMicroseconds(4800); // Approx 200Hz
  }

  // 2. PROCESS (Calculate features)
  // We need distinct arrays for FFT because the library modifies the input array
  // So we calculate RMS/Kurtosis BEFORE FFT or use copies. 
  // Ideally, calculate statistical features first (non-destructive):
  float rmsX = calcRMS(vRealX);
  float rmsY = calcRMS(vRealY);
  float rmsZ = calcRMS(vRealZ);
  float kurtZ = calcKurtosis(vRealZ);

  // Now do FFT (Destructive to vReal arrays)
  double peakX = getPeakFreq(vRealX, vImagX);
  double peakY = getPeakFreq(vRealY, vImagY);
  double peakZ = getPeakFreq(vRealZ, vImagZ);
  
  // 3. PACK (Single JSON Object)
  String json = "{";
  json += "\"rms_x\":" + String(rmsX, 4) + ",";
  json += "\"rms_y\":" + String(rmsY, 4) + ",";
  json += "\"rms_z\":" + String(rmsZ, 4) + ",";
  json += "\"peak_freq_x\":" + String(peakX, 2) + ",";
  json += "\"peak_freq_y\":" + String(peakY, 2) + ",";
  json += "\"peak_freq_z\":" + String(peakZ, 2) + ",";
  json += "\"kurtosis_z\":" + String(kurtZ, 4);
  json += "}";

  Serial.println("Sending: " + json);

  // 4. SEND (Immediate)
  sendData(json);

  // 5. TIMING
  // Sampling ~600ms + Calc ~50ms + Net ~100ms = ~750ms.
  // Add 250ms delay to approximate 1 second loop.
  delay(250); 
}

// --- Helpers ---

// Helper to run FFT on specific arrays
double getPeakFreq(double* vReal, double* vImag) {
    ArduinoFFT<double> FFT = ArduinoFFT<double>(vReal, vImag, samples, samplingFrequency);
    FFT.windowing(FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.compute(FFT_FORWARD);
    FFT.complexToMagnitude();
    return FFT.majorPeak();
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
  var /= samples; 
  fourth /= samples;
  
  if (var < 0.0001) return 0;
  return (fourth / (var*var)) - 3.0;
}

void sendData(String payload) {
  if(WiFi.status() != WL_CONNECTED) return;
  
  WiFiClient client;
  HTTPClient http;
  
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "application/json");
  
  int code = http.POST(payload);
  
  if (code > 0) {
      String response = http.getString();
      Serial.println(">> Server Response: " + response);
  } else {
      Serial.printf(">> Error: %s\n", http.errorToString(code).c_str());
  }
  
  http.end();
}