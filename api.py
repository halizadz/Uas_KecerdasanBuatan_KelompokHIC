from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib
import logging
from flask_cors import CORS 
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis
from text_to_score import text_to_score


app = Flask(__name__)
CORS(app) 

# Konfigurasi Redis untuk Limiter
redis_client = Redis(host='redis', port=6379, decode_responses=True)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="redis://redis:6379",
    default_limits=["1000 per day", "100 per hour"]
)

@app.route('/')
def home():
    return jsonify({
        "message": "ANFIS-Fuzzy Criticality Analysis System API",
        "endpoints": {
            "/predict": "POST - Untuk prediksi criticality",
            "/health": "GET - Untuk pengecekan server"
        }
    })

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model ANFIS
class CustomANFIS(nn.Module):
    def __init__(self, n_input, n_rules):
        super().__init__()
        self.n_input = n_input
        self.n_rules = n_rules
        
        # Parameter untuk membership function (Gaussian)
        self.mu = nn.Parameter(torch.rand(n_input, n_rules))  # mean
        self.sigma = nn.Parameter(torch.rand(n_input, n_rules))  # std dev
        
        # Layer konsekuen (linear)
        self.consequent = nn.Linear(n_input * n_rules, 1, bias=False)
    
    def forward(self, x):
        # Hitung membership values (Gaussian MF)
        x = x.unsqueeze(-1).expand(-1, -1, self.n_rules)
        mf = torch.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
        
        # Hitung firing strength (product rule)
        strength = torch.prod(mf, dim=1)
        strength = strength / (strength.sum(dim=1, keepdim=True) + 1e-12)
        
        # Hitung output
        x_rep = x.reshape(x.shape[0], -1)
        consequent_out = self.consequent(x_rep)
        return torch.sum(strength * consequent_out, dim=1).unsqueeze(-1)

# Load model dan scalers
try:
    model = CustomANFIS(n_input=5, n_rules=5)
    model.load_state_dict(torch.load('anfis_model.pth'))
    model.eval()  # Set model ke mode evaluasi
    
    scaler_X = joblib.load('scaler_X.save') 
    scaler_y = joblib.load('scaler_y.save')
    
    logger.info("Model dan scaler berhasil dimuat")
except Exception as e:
    logger.error(f"Gagal memuat model atau scaler: {str(e)}")
    raise

# Threshold criticality
BETA_CR = 0.477

def preprocess_input(data):
    processed = {}
    for field in ['scope', 'prospects', 'potential', 'economy', 'efficiency']:
        value = data.get(field, 0.5)
        if isinstance(value, (list, tuple)):
            # Handle text descriptions
            processed[field] = text_to_score(" ".join(value), field)
        elif isinstance(value, (int, float)):
            # Direct numeric value
            processed[field] = float(value)
        else:
            # Default fallback
            processed[field] = 0.5
    return processed


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    try:
        data = request.json
        logger.info(f"DATA MASUK: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Validate input structure
        if not isinstance(data, dict):
            return jsonify({'error': 'Input must be a JSON object'}), 400
            
        # Process input
        try:
            processed_data = preprocess_input(data)
            X = np.array([[
                float(processed_data['scope']),
                float(processed_data['prospects']),
                float(processed_data['potential']),
                float(processed_data['economy']),
                float(processed_data['efficiency'])
            ]])
        except (ValueError, KeyError) as e:
            return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
            
        # Validasi range input (0-1)
        if not all(0 <= x <= 1 for x in X[0]):
            return jsonify({'error': 'Input values must be between 0 and 1'}), 400
            
        # Scaling dan prediksi
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(X_tensor)
            prediction = scaler_y.inverse_transform(prediction.numpy())
        
        criticality = float(prediction[0][0])
        status = 'Critical' if criticality >= BETA_CR else 'Non-Critical'
        
        return jsonify({
            'criticality': criticality,
            'status': status,
            'input_scores': processed_data  # ✅ Ini bagian yang diperlukan frontend
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api-docs')
def api_docs():
    return jsonify({
        "description": "API Documentation for ANFIS-Fuzzy Criticality Analysis",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict technology criticality",
                "parameters": {
                    "scope": "float [0-1]",
                    "prospects": "float [0-1]",
                    "potential": "float [0-1]",
                    "economy": "float [0-1]",
                    "efficiency": "float [0-1]"
                },
                "example_request": {
                    "scope": 0.7,
                    "prospects": 0.8,
                    "potential": 0.6,
                    "economy": 0.5,
                    "efficiency": 0.9
                }
            }
        }
    })

@app.route('/analysis', methods=['GET'])
@limiter.exempt
def analysis():
    return jsonify({'message': 'Analysis endpoint works. Implement as needed.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)