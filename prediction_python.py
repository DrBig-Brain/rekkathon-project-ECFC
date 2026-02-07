from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 1. Load the Model
# Make sure 'motor_model.pkl' is in the same folder as this script
try:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    print(">> Model loaded successfully!")
except FileNotFoundError:
    print(">> ERROR: 'motor_model.pkl' not found.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Receive Data
        data = request.get_json()
        
        # 3. Format Data
        # IMPORTANT: The order here MUST match the order used during training
        features = [
            data['rms_x'],
            data['rms_y'],
            data['rms_z'],
            data['peak_freq_x'],
            data['peak_freq_y'],
            data['peak_freq_z'],
            data['kurtosis_z']
        ]
        
        # Convert to DataFrame (Sklearn often prefers this over raw arrays to silence warnings)
        feature_names = ['rms_x', 'rms_y', 'rms_z', 'peak_freq_x', 'peak_freq_y', 'peak_freq_z', 'kurtosis_z']
        input_df = pd.DataFrame([features], columns=feature_names)

        # 4. Predict
        prediction = model.predict(input_df)[0]
        
        print(f"Received Data: {features} -> Prediction: {prediction}")

        # 5. Reply to ESP
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    # '0.0.0.0' makes the server accessible to other devices on your WiFi
    app.run(host='0.0.0.0', port=5000)