from flask import Flask, request, jsonify, render_template, url_for
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("C:\\Users\\LENOVO\\Desktop\\project\\save_model", "model.joblib")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)
print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predictions', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in request JSON'}), 400

        features = np.array(data['features']).reshape(1, -1)  # Reshape for model input
        expected_feature_size = 150528  # Expected input feature size

        if features.shape[1] != expected_feature_size:
            return jsonify({'error': f'Expected input feature size ({expected_feature_size}), got {features.shape[1]}'}), 400

        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})  # Convert to Python native type
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluationmetrics')
def evaluation_metrics():
    return render_template('evaluationmetrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

if __name__ == '__main__':
    app.run(debug=True)
