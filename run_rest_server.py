# USAGE
# Start the server:
#       python3 run_keras_server.py
# Submit a request via cURL:
#       curl -X POST -F image=@dog.jpg 'https://telemir.de:32770/predict'

import os
import io
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug import secure_filename
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from PIL import Image

# initialize our Flask application and the Keras model
app = Flask(__name__)

# globale Variables
model = None
img_width, img_height = 299, 299

def load_trained_model():
    global model
    model_path = './models/model.h5'
    model_weights_path = './models/weights.h5'
    model = load_model(model_path)
    model._make_predict_function()
    model.load_weights(model_weights_path)
    print("* Keras: Das Modell wurde erfolgreich geladen :)")

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(img_width, img_height))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = preds.tolist() # Für jsonify ein Array to List konvertieren
            data["predictions"] = []
            
            labels = [ 'Malamute', 'Pitbull', 'Retriever']
            # loop over the results and add them to the list of
            # returned predictions
            r = {"label": labels[0], "probability": results[0][0]}
            data["predictions"].append(r)
            r = {"label": labels[1], "probability": results[0][1]}
            data["predictions"].append(r)
            r = {"label": labels[2], "probability": results[0][2]}
            data["predictions"].append(r)
          
            if np.argmax(preds) == 0:
                # Alaskan Malamute
                data["decision"] = "Alaskan Malamute" 
                data["success"] = True
            elif np.argmax(preds) == 1:
                data["decision"] = "Pitbull" 
                data["success"] = True
            elif np.argmax(preds) == 2:
                data["decision"] = "Golden Retriever" 
                data["success"] = True
            else:
                print("Fehler: ein komisches Ergebnis von predict erhalten!")
                data["decision"] = "Unknown" 
                data["success"] = False

    # return the data dictionary as a JSON response
    return jsonify(data)

# start the server
if __name__ == "__main__":
        print(("* Loading Keras model and Flask starting server..."
                "please wait until server has fully started"))
        load_trained_model()
        app.debug=True
        app.run(host='0.0.0.0', port=32770, ssl_context=('ssl/fullchain.pem', 'ssl/privkey.pem'))
