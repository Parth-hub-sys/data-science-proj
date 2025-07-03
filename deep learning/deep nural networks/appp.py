from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('concrete_strength_model.h5', compile=False)
scaler = joblib.load('scaler.save')  # Save your scaler after fitting

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expecting a list of 8 features
    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return jsonify({'prediction': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
    