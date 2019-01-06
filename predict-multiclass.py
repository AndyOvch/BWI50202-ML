#Usage: python predict-multiclass.py
#https://github.com/tatsuyah/CNN-Image-Classifier

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
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

malamute_t = 0
malamute_f = 0
pitbull_t = 0
pitbull_f = 0
retriever_t = 0
retriever_f = 0

for i, ret in enumerate(os.walk('./test-data/Malamut')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Malamute")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      print(ret[0] + '/' + filename)
      malamute_t += 1
    else:
      malamute_f += 1

for i, ret in enumerate(os.walk('./test-data/Pitbull')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Pitbull")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      print(ret[0] + '/' + filename)
      pitbull_t += 1
    else:
      pitbull_f += 1

for i, ret in enumerate(os.walk('./test-data/Retriever')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: Retriever")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      print(ret[0] + '/' + filename)
      retriever_t += 1
    else:
      retriever_f += 1

"""
Check metrics
"""
print("True Malamute: ", malamute_t)
print("False Malamute: ", malamute_f)
print("True Pitbull: ", pitbull_t)
print("False Pitbull: ", pitbull_f)
print("True Retriever: ", retriever_t)
print("False Retriever: ", retriever_f)
