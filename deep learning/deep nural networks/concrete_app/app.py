from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os
import pandas as pd
# Load model and scaler
model = tf.keras.models.load_model('./model/concrete_strength_model.h5', compile=False)
scaler = joblib.load('./model/scaler.save')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[f'feature{i}']) for i in range(1, 9)]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)
        result = round(prediction[0][0], 2)
        return jsonify({'prediction': f'{result} MPa'})
    except Exception as e:
        return jsonify({'error': str(e)})
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        print("Batch predict endpoint hit!")  # Add this line
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        file = request.files['file']
        df = pd.read_csv(file)
        print(df.head())  # Add this line
        if df.shape[1] != 8:
            return jsonify({'error': 'CSV must have 8 columns'})
        scaled = scaler.transform(df.values)
        preds = model.predict(scaled)
        preds = [round(float(p[0]), 2) for p in preds]
        return jsonify({'predictions': preds})
    except Exception as e:
        print("Error:", e)  # Add this line
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
