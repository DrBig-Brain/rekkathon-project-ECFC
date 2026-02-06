from flask import Flask, request, jsonify
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== CONFIGURATION ====================
MODEL_PATH = 'anomaly_model.pkl'
SCALER_PATH = 'scaler.pkl'
TRAINING_DATA_PATH = 'training_data.csv'
PREDICTIONS_LOG = 'predictions.csv'

# Training parameters
TRAINING_BUFFER_SIZE = 100  # Collect 100 samples before training
RETRAIN_THRESHOLD = 200     # Retrain every 200 new samples

# Anomaly detection threshold
ANOMALY_THRESHOLD = -0.3    # Lower = stricter anomaly detection

# ==================== GLOBAL STATE ====================
training_buffer = []
sample_count = 0
model = None
scaler = None

# Load existing model if available
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✓ Loaded existing model from disk")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
        scaler = None
else:
    print("⚠ No existing model found - will train when data arrives")

# ==================== HELPER FUNCTIONS ====================

def extract_features(data_point):
    """Extract features from a single data point"""
    features = [
        data_point.get('peak_freq_x', 0),
        data_point.get('peak_freq_y', 0),
        data_point.get('peak_freq_z', 0),
        data_point.get('rms_x', 0),
        data_point.get('rms_y', 0),
        data_point.get('rms_z', 0),
        data_point.get('kurtosis', 0),
    ]
    
    # Add spectrum features (top 5 magnitudes for each axis)
    spectrum_x = data_point.get('spectrum_x', [0]*5)
    spectrum_y = data_point.get('spectrum_y', [0]*5)
    spectrum_z = data_point.get('spectrum_z', [0]*5)
    
    features.extend(spectrum_x)
    features.extend(spectrum_y)
    features.extend(spectrum_z)
    
    return np.array(features).reshape(1, -1)

def train_model(data_points):
    """Train the anomaly detection model"""
    global model, scaler
    
    print(f"\n{'='*50}")
    print(f"🔧 TRAINING MODEL with {len(data_points)} samples")
    print(f"{'='*50}")
    
    # Extract features from all data points
    feature_list = []
    for point in data_points:
        features = extract_features(point)
        feature_list.append(features.flatten())
    
    X = np.array(feature_list)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature ranges:")
    print(f"  Min: {X.min(axis=0)[:3]}")
    print(f"  Max: {X.max(axis=0)[:3]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.1,      # Expect 10% anomalies
        n_estimators=100,
        max_samples='auto',
        random_state=42,
        verbose=0
    )
    
    model.fit(X_scaled)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save training data
    df = pd.DataFrame(feature_list)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    
    print(f"✓ Model trained and saved successfully!")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Scaler: {SCALER_PATH}")
    print(f"  Data: {TRAINING_DATA_PATH}")
    print(f"{'='*50}\n")

def predict_anomaly(data_point):
    """Predict if a data point is anomalous"""
    global model, scaler
    
    if model is None or scaler is None:
        return None, None, "Model not trained yet"
    
    # Extract and scale features
    features = extract_features(data_point)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]  # 1 = normal, -1 = anomaly
    score = model.score_samples(features_scaled)[0]  # Anomaly score
    
    is_anomaly = prediction == -1 or score < ANOMALY_THRESHOLD
    
    return is_anomaly, score, None

def log_prediction(data_point, is_anomaly, score):
    """Log predictions to CSV for analysis"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        'timestamp': timestamp,
        'is_anomaly': is_anomaly,
        'anomaly_score': score,
        'peak_freq_x': data_point.get('peak_freq_x', 0),
        'peak_freq_y': data_point.get('peak_freq_y', 0),
        'peak_freq_z': data_point.get('peak_freq_z', 0),
        'rms_x': data_point.get('rms_x', 0),
        'rms_y': data_point.get('rms_y', 0),
        'rms_z': data_point.get('rms_z', 0),
    }
    
    df = pd.DataFrame([log_entry])
    
    # Append to CSV
    if os.path.exists(PREDICTIONS_LOG):
        df.to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(PREDICTIONS_LOG, mode='w', header=True, index=False)

# ==================== FLASK ROUTES ====================

@app.route('/data', methods=['POST'])
def receive_data():
    global training_buffer, sample_count
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    if not request.is_json:
        print(f"[{timestamp}] ✗ Received non-JSON request")
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400
    
    try:
        batch_data = request.get_json()
        
        # Handle both single object and array
        if isinstance(batch_data, dict):
            batch_data = [batch_data]
        
        if not isinstance(batch_data, list):
            return jsonify({"status": "error", "message": "Expected JSON array or object"}), 400
        
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 📦 BATCH RECEIVED: {len(batch_data)} samples")
        print(f"{'='*60}")
        
        results = []
        
        for idx, data_point in enumerate(batch_data):
            mode = data_point.get('mode', 'training')
            device_time = data_point.get('timestamp', 0)
            
            print(f"\n--- Sample {idx + 1}/{len(batch_data)} ---")
            print(f"Mode: {mode.upper()}")
            print(f"Device Time: {device_time} ms")
            print(f"Peak Frequencies: X={data_point.get('peak_freq_x', 0):.2f} Hz, "
                  f"Y={data_point.get('peak_freq_y', 0):.2f} Hz, "
                  f"Z={data_point.get('peak_freq_z', 0):.2f} Hz")
            print(f"RMS Values: X={data_point.get('rms_x', 0):.4f}, "
                  f"Y={data_point.get('rms_y', 0):.4f}, "
                  f"Z={data_point.get('rms_z', 0):.4f}")
            print(f"Kurtosis: {data_point.get('kurtosis', 0):.4f}")
            
            if mode == 'training':
                # Add to training buffer
                training_buffer.append(data_point)
                sample_count += 1
                
                print(f"📊 Training buffer: {len(training_buffer)}/{TRAINING_BUFFER_SIZE}")
                
                # Train when buffer is full
                if len(training_buffer) >= TRAINING_BUFFER_SIZE:
                    train_model(training_buffer)
                    training_buffer = []  # Clear buffer after training
                
                results.append({
                    "sample": idx + 1,
                    "mode": "training",
                    "status": "buffered",
                    "buffer_size": len(training_buffer)
                })
            
            elif mode == 'prediction':
                # Predict anomaly
                is_anomaly, score, error = predict_anomaly(data_point)
                
                if error:
                    print(f"⚠️  {error}")
                    results.append({
                        "sample": idx + 1,
                        "mode": "prediction",
                        "status": "error",
                        "message": error
                    })
                else:
                    # Log prediction
                    log_prediction(data_point, is_anomaly, score)
                    
                    if is_anomaly:
                        print(f"🚨 ANOMALY DETECTED!")
                        print(f"   Anomaly Score: {score:.4f}")
                        print(f"   Threshold: {ANOMALY_THRESHOLD}")
                    else:
                        print(f"✓ Normal operation")
                        print(f"   Anomaly Score: {score:.4f}")
                    
                    results.append({
                        "sample": idx + 1,
                        "mode": "prediction",
                        "status": "normal" if not is_anomaly else "anomaly",
                        "anomaly_score": float(score),
                        "is_anomaly": bool(is_anomaly)
                    })
        
        print(f"\n{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "timestamp": timestamp,
            "samples_processed": len(batch_data),
            "results": results
        }), 200
        
    except Exception as e:
        print(f"[{timestamp}] ✗ Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
        "model_trained": model is not None,
        "training_buffer_size": len(training_buffer),
        "total_samples_received": sample_count,
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        "predictions_logged": os.path.exists(PREDICTIONS_LOG)
    }), 200

@app.route('/reset', methods=['POST'])
def reset_model():
    """Reset the model and training data"""
    global model, scaler, training_buffer, sample_count
    
    try:
        # Remove files
        for file in [MODEL_PATH, SCALER_PATH, TRAINING_DATA_PATH]:
            if os.path.exists(file):
                os.remove(file)
        
        # Reset state
        model = None
        scaler = None
        training_buffer = []
        sample_count = 0
        
        print("\n🔄 Model and training data reset\n")
        
        return jsonify({"status": "success", "message": "Model reset successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================== STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Industrial Anomaly Detection Server Starting...")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Scaler Path: {SCALER_PATH}")
    print(f"Training Data: {TRAINING_DATA_PATH}")
    print(f"Predictions Log: {PREDICTIONS_LOG}")
    print(f"\nEndpoints:")
    print(f"  POST /data   - Receive sensor data")
    print(f"  GET  /status - Check system status")
    print(f"  POST /reset  - Reset model and data")
    print(f"\nTraining Parameters:")
    print(f"  Buffer Size: {TRAINING_BUFFER_SIZE} samples")
    print(f"  Anomaly Threshold: {ANOMALY_THRESHOLD}")
    print("="*60 + "\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)