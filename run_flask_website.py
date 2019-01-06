#Usage: python3 run_flask_website.py
import os
 
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename, SharedDataMiddleware
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64

img_width, img_height = 299, 299
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model._make_predict_function()
model.load_weights(model_weights_path)
print("Keras: Das Modell wurde erfolgreich geladen :)")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: Alaskan Malamute")
    elif answer == 1:
        print("Label: Pitbull")
    elif answer == 2:
        print("Label: Golden Retriever")
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../static/template.png')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = 'Alaskan Malamute'
            elif result == 1:
                label = 'Pitbull'			
            elif result == 2:
                label = 'Golden Retriever'
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)
        else:
            print("Wrong filetype detected! Reloading template...")
            return render_template('template.html', label='', imagesource='../static/template.png')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='127.0.0.1', port=32769)
