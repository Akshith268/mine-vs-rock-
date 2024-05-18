from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found in the current directory.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Sonar Model Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)

        print(f"Received features: {features}")
        print(f"Expected number of features: {model.n_features_in_}")

        if features.shape[1] != model.n_features_in_:
            return jsonify({'error': f'Input data must have {model.n_features_in_} features.'}), 400
        
        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
