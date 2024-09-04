import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request,url_for
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image  # Import the Image module

ensemble_model = tf.keras.models.load_model('brain_ensemble.h5')


app = Flask(__name__)


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/uploads/" + imagefile.filename
    imagefile.save(image_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img = img.resize((224, 224))  # Resize the image to match the model's input size
    img = np.array(img)  # Convert to array
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = ensemble_model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Replace with your actual class labels

    # Compute confidence scores
    confidence_scores = prediction[0]  # Get the probability distribution for the image
    confidence = confidence_scores[predicted_class[0]] * 100   # Get the confidence for the predicted class

    classification = class_labels[predicted_class[0]]

    print(f'Predicted class: {classification}')
    print(f'Confidence score: {confidence:.2f}')

    # Return result to the template
    return render_template('index.html', pr=True, prediction=classification, confidence=f"{confidence:.2f}", image_name=imagefile.filename)




if __name__ == '__main__':
    # app.run(port=0.0.0.0,debug=True)
    app.run(host='0.0.0.0',debug=True)
