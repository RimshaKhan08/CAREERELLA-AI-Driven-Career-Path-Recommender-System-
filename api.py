from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, static_folder='static')  # put your HTML, CSS, JS in 'static'

# ------------------- Load model and metadata -------------------
model = load_model('career_recommender_model.h5')

with open("feature_map.pkl", "rb") as f:
    feature_map = pickle.load(f)

with open("career_columns.pkl", "rb") as f:
    career_columns = pickle.load(f)

print("Model expects:", model.input_shape)
print("Number of features in feature map:", len(feature_map))

# ------------------- Serve HTML page -------------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ------------------- Prediction API -------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_array = np.zeros(len(feature_map))  # 23 features

    # Map input values to the correct feature indices
    for col, val in data.items():
        if isinstance(val, list):  # for Skills or Interests with multiple values
            for v in val:
                key = f"{col}_{v.lower()}" if f"{col}_{v.lower()}" in feature_map else v.lower()
                if key in feature_map:
                    input_array[feature_map[key]] = 1
        else:  # single value features
            key = f"{col}_{val.lower()}" if f"{col}_{val.lower()}" in feature_map else val.lower()
            if key in feature_map:
                input_array[feature_map[key]] = 1

    input_array = input_array.reshape(1, -1)  # shape (1, 23)
    pred_prob = model.predict(input_array)
    pred_class = career_columns[np.argmax(pred_prob)]

    return jsonify({
        'predicted_career': pred_class,
        'probabilities': dict(zip(career_columns, pred_prob[0].tolist()))
    })

# ------------------- Run Flask app -------------------
if __name__ == "__main__":
    app.run(debug=True)
