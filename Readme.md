# MotorSense AI

MotorSense AI is an IoT-based system designed to monitor motor health using machine learning. It collects vibration data using an ESP8266/ESP32 microcontroller and an MPU6050 accelerometer, analyzes the data to detect anomalies, and classifies the motor's state (e.g., "Good", "Bad").

## Features

- **Real-time Data Collection**: Captures accelerometer input (RMS, Peak Frequency, Kurtosis) via an ESP microcontroller.
- **Machine Learning Classification**: Uses a trained model (Random Forest) to classify motor health states.
- **Web Dashboard/API**: Flask-based backend for logging data and serving predictions.
- **Edge Integration**: Microcontroller collects data and sends it to the central server for inference.

## Project Structure

```
├── data_extraction/          # Arduino sketch for data collection
│   └── data_extraction.ino
├── prediction/               # Arduino sketch for real-time prediction
│   └── prediction.ino
├── data_extraction.py        # Flask server for data logging
├── prediction_python.py      # Flask server for real-time inference
├── model.ipynb               # Jupyter Notebook for model training
├── model.pkl                 # Trained ML model
├── training_data_good.csv    # Training dataset (Good state)
├── training_data_bad.csv     # Training dataset (Bad state)
└── Readme.md                 # Project documentation
```

## Hardware Requirements

- **Microcontroller**: ESP8266 or ESP32
- **Sensor**: MPU6050 Accelerometer/Gyroscope
- **Connection**: WiFi network for communication between the microcontroller and the server

## Software Requirements

- Python 3.x
- Arduino IDE (with ESP8266/ESP32 board support)

### Python Dependencies
Install the required libraries:

```bash
pip install flask pandas scikit-learn
```

### Arduino Libraries
Install these libraries via the Arduino Library Manager:
- `Adafruit MPU6050`
- `ArduinoFFT`
- `ESP8266WiFi` / `ESP8266HTTPClient` (Built-in for ESP8266 core)

## Setup & Configuration

### 1. Hardware Connection
Connect the MPU6050 to the ESP microcontroller:
- **VCC** -> 3.3V / 5V
- **GND** -> GND
- **SDA** -> GPIO 4 (D2 on NodeMCU)
- **SCL** -> GPIO 5 (D1 on NodeMCU)

*(Pin mapping may vary based on your specific ESP board)*

### 2. Server Configuration
Find your computer's local IP address (e.g., run `ipconfig` or `ifconfig`).

### 3. Arduino Configuration
Open `data_extraction/data_extraction.ino` and `prediction/prediction.ino` and update the following lines:

```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverUrl = "http://YOUR_LOCAL_IP:5000/log"; // For data extraction
// OR
const char* serverUrl = "http://YOUR_LOCAL_IP:5000/predict"; // For prediction
```

## Usage

### Phase 1: Data Collection
1.  Start the logging server:
    ```bash
    python data_extraction.py
    ```
2.  Upload `data_extraction/data_extraction.ino` to the ESP.
3.  The ESP will gather data and send it to the server. The data gets saved to `training_data.csv`.
4.  Collect data for different motor states (e.g., normal operation, induced faults). *Note: You might need to manually rename the output CSV or manage different files for different labels.*

### Phase 2: Model Training
1.  Open `model.ipynb` in multipleter/VS Code.
2.  Load your collected CSV files (`training_data_good.csv`, `training_data_bad.csv`, etc.).
3.  Run the cells to train the Random Forest model.
4.  The trained model will be saved as `model.pkl`.

### Phase 3: Real-time Prediction
1.  Start the prediction server:
    ```bash
    python prediction_python.py
    ```
2.  Upload `prediction/prediction.ino` to the ESP.
3.  The ESP will send real-time vibration data to the server.
4.  The server will print the prediction ("Good" or "Bad") to the console and return it to the ESP.

## Troubleshooting
- **Connection Failed**: Ensure both the ESP and your computer are on the same WiFi network. Check if your firewall is blocking port 5000.
- **MPU Connection**: Check wiring and I2C addresses. Use an I2C scanner sketch to verify the sensor is detected.
