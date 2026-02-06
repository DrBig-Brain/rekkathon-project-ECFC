from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# This file will store your training data
CSV_FILE = 'training_data.csv'

@app.route('/log', methods=['POST'])
def log_data():
    try:
        # 1. Get the batch
        batch_data = request.get_json()
        if not batch_data or not isinstance(batch_data, list):
            return jsonify({"error": "Invalid format"}), 400

        # 2. Add server-side timestamp (optional but useful)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for entry in batch_data:
            entry['server_time'] = timestamp

        # 3. Convert to DataFrame
        df = pd.DataFrame(batch_data)

        # 4. Append to CSV
        # If file doesn't exist, write header. If it does, skip header.
        header = not os.path.exists(CSV_FILE)
        df.to_csv(CSV_FILE, mode='a', header=header, index=False)

        print(f"[{timestamp}] Saved {len(batch_data)} samples. Label: {batch_data[0].get('label', 'unknown')}")
        
        return jsonify({"status": "saved", "count": len(batch_data)}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Makes the server accessible on your local network
    app.run(host='0.0.0.0', port=5000)