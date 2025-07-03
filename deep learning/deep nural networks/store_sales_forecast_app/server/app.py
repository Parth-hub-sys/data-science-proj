from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='../client', static_url_path='')

# Load model (placeholder)
model_path = '../model/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    store_nbr = data.get('store_nbr')
    family = data.get('family')
    days = int(data.get('days'))

    # Dummy prediction logic (replace with real model)
    forecast = [{'date': f'Day {i+1}', 'sales': float(np.random.uniform(20, 50))} for i in range(days)]

    return jsonify({'forecast': forecast})

if __name__ == '__main__':
    app.run(debug=True)