from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Make sure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = tf.keras.models.load_model('model/fashion_model2.h5')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    label = None
    confidence = None
    error = None

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Open and preprocess image
                img = Image.open(filepath).convert('L').resize((28, 28))
                img_array = np.array(img) / 255.0
                img_array = img_array.reshape(1, 28, 28)

                # Predict
                predictions = model.predict(img_array)
                pred_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                label = class_names[pred_index]
            except UnidentifiedImageError:
                error = "Uploaded file is not a valid image."
        else:
            error = "Invalid file type. Please upload a .png, .jpg or .jpeg image."

    return render_template('index.html',
                           filename=filename,
                           label=label,
                           confidence=confidence,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)
