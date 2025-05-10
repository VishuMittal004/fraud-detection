from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load the pre-trained model and data
def load_saved_model():
    with open('fraud_model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data['model'], saved_data['scaler'], saved_data['feature_columns']

model, scaler, feature_columns = load_saved_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        amount = float(data['amount'])
        date_time = datetime.strptime(f"{data['date']} {data['time']}", "%Y-%m-%d %H:%M")
        
        # Create feature vector matching the training data structure
        features = np.zeros(len(feature_columns))
        feature_dict = dict(zip(feature_columns, range(len(feature_columns))))
        
        # Set Time feature (seconds since midnight)
        seconds_since_midnight = (date_time.hour * 3600 + date_time.minute * 60)
        features[feature_dict['Time']] = seconds_since_midnight
        
        # Set Amount feature
        features[feature_dict['Amount']] = amount
        
        # Scale features using the saved scaler
        features_scaled = scaler.transform([features])
        
        # Make prediction using the saved model
        prediction = model.predict_proba(features_scaled)[0]
        fraud_probability = float(prediction[1] * 100)
        
        return jsonify({
            'fraud_probability': round(fraud_probability, 2),
            'is_fraud': bool(fraud_probability > 50)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
