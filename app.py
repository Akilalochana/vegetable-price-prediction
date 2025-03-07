from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Global dictionaries to store models and scalers
models = {}
scalers = {}

def load_all_models():
    """Load all saved models and scalers at startup"""
    print("Loading models...")
    
    # Get all model files in the directory
    model_files = [f for f in os.listdir('saved_models') if f.endswith('_model.joblib')]
    commodities = [f.split('_model.joblib')[0] for f in model_files]
    
    print(f"Found models for: {commodities}")
    
    for commodity in commodities:
        try:
            model_path = f'saved_models/{commodity}_model.joblib'
            scaler_path = f'saved_models/{commodity}_scaler.joblib'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models[commodity] = joblib.load(model_path)
                scalers[commodity] = joblib.load(scaler_path)
                print(f"Loaded model for {commodity}")
            else:
                print(f"Missing files for {commodity}")
        except Exception as e:
            print(f"Error loading model for {commodity}: {e}")

# Test endpoint
@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        commodity = data['commodity']
        
        if commodity not in models:
            return jsonify({
                'error': f'No model found for {commodity}'
            }), 404
            
        # Get current date features
        current_date = datetime.now()
        features = {
            'Month': current_date.month,
            'Year': current_date.year,
            'Month_sin': np.sin(2 * np.pi * current_date.month/12),
            'Month_cos': np.cos(2 * np.pi * current_date.month/12),
            'Quarter': (current_date.month-1)//3 + 1
        }
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Scale features
        X_scaled = scalers[commodity].transform(X)
        
        # Make prediction
        prediction = models[commodity].predict(X_scaled)[0]
        
        # Get predictions for next 2 months
        predictions = []
        for i in range(3):  # Current month + 2 future months
            month = current_date.month + i
            year = current_date.year
            if month > 12:
                month -= 12
                year += 1
                
            features['Month'] = month
            features['Year'] = year
            features['Month_sin'] = np.sin(2 * np.pi * month/12)
            features['Month_cos'] = np.cos(2 * np.pi * month/12)
            features['Quarter'] = (month-1)//3 + 1
            
            X = pd.DataFrame([features])
            X_scaled = scalers[commodity].transform(X)
            pred = models[commodity].predict(X_scaled)[0]
            predictions.append({
                'month': month,
                'year': year,
                'price': round(pred, 2)
            })
        
        return jsonify({
            'commodity': commodity,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_all_models()
    app.run(debug=True)

# Load models when starting the app
load_all_models()