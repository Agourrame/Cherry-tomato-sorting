from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image(image_path):
    """Loads an image from the given path and preprocesses it."""
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize images to 100x100 pixels
    img = img.convert('RGB')  # Convert to RGB format
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            img = load_image(file_path)
            model = tf.keras.models.load_model('app/models/tomato_sorter_model.keras') 
            prediction = model.predict(np.expand_dims(img, axis=0))
            predicted_class = np.argmax(prediction)
            class_labels = ['ripe', 'semi-ripe', 'unripe']
            predicted_label = class_labels[predicted_class]

            return jsonify({'prediction': predicted_label})
        except FileNotFoundError:
            return jsonify({'error': 'File not found.'})
        except IOError:
            return jsonify({'error': 'Error reading the image.'})
        except tf.errors.InvalidArgumentError:
            return jsonify({'error': 'Error with image processing.'})
        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid file type'})

if __name__ == "__main__":
    app.run(debug=True) 