#Usage: python predict-multiclass.py
#https://github.com/tatsuyah/CNN-Image-Classifier

import os
import sys
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 299, 299
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
print("Keras: Das Model wurde erfolgreich geladen.")

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  array = model.predict(x, verbose=1)
  result = array[0]
  answer = np.argmax(result)

  if answer == 0:
    print("Label: Alaskan Malamute")
  elif answer == 1:
    print("Label: Pitbull")
  elif answer == 2:
    print("Label: Golden Retriever")

  return answer

print("Ergebnis: ", predict(sys.argv[1]))
