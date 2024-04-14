from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import pickle

app = Flask(__name__)

# Load the image classification model
image_model = load_model('model.h5')

# Load the chronic kidney disease prediction model
ckd_model = pickle.load(open('kidney_svm_final.sav', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST', 'GET'])
def predict_image():
    if request.method == 'GET':
        return render_template('predict_image.html')
    elif request.method == 'POST':
        file = request.files['file']

        # Open image file and convert to RGB
        img = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Resize image to match model input size
        img = img.resize((224, 224))

        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        prediction = image_model.predict(img_array)
        predicted_class = np.argmax(prediction)

        class_labels = {0: 'Kidney Stone', 1: 'Normal'}  # Update with your class labels
        predicted_label = class_labels[predicted_class]

        return jsonify({'prediction': predicted_label})  # Send the prediction result as JSON


@app.route('/predict_ckd', methods=['POST', 'GET'])
def predict_ckd():
    if request.method == 'GET':
        return render_template('predict_ckd.html')
    elif request.method == 'POST':
        data = request.get_json()['data']
        input_data = np.array(data).reshape(1, -1)
        prediction = ckd_model.predict(input_data)

        if prediction[0] == 0:
            result = 'The Person may have a Kidney Disease, Please consult a doctor!'
        else:
            result = "The Person's kidney is Normal"

        return result  # Returning just the string prediction


if __name__ == '__main__':
    app.run(debug=True)
