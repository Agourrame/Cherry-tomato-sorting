from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('outputs/models/tomato_sorter.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = os.path.join('static', 'uploaded_images', file.filename)
    file.save(file_path)

    # Preprocess the image
    image = cv2.imread(file_path)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    color = np.argmax(predictions[0])  # Classification
    diameter = predictions[1][0][0]   # Regression

    # Map results
    color_map = {0: 'Unripe (Green)', 1: 'Semi-ripe (Orange)', 2: 'Ripe (Red)'}
    result = {'color': color_map[color], 'diameter': f"{round(diameter, 2)} mm"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
