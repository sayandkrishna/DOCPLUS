from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization

app = Flask(__name__)


# Custom BatchNormalization class to handle the axis issue
class CustomBatchNormalization(BatchNormalization):
    def __init__(self, axis=3, **kwargs):
        if isinstance(axis, list):
            axis = axis[0]  # Convert list to int
        super().__init__(axis=axis, **kwargs)

    @classmethod
    def from_config(cls, config):
        if isinstance(config.get('axis'), list):
            config['axis'] = config['axis'][0]  # Convert axis from list to int
        return super().from_config(config)


# Load your trained model with the custom BatchNormalization deserialization
model = tf.keras.models.load_model('models/my_model.h5',
                                   custom_objects={'BatchNormalization': CustomBatchNormalization})


# Define the prediction function
def predict_tumor(img_path):
    img = Image.open(img_path).resize((299, 299))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Update with your class labels
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define the route for handling image uploads and predictions
@app.route('/', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img_path = 'uploads/' + file.filename  # Save the uploaded image temporarily
        file.save(img_path)
        prediction = predict_tumor(img_path)
        return render_template('bt.html', prediction=prediction, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
