# Usage: python3 predict-test-dir-IResNetV2.py [path_to_directory_with_pictures] [correct_class]
# For instance: "python3 predict-test-dir-IResNetV2.py ./data/validation/Pitbull 1" 

import os
import sys
import argparse
import numpy as np

from argparse import RawTextHelpFormatter
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 299, 299
breeds = ("Alaskan Malamute", "Pitbull", "Golden Retriever")
counter_ok, counter_wrong = 0, 0

model_path = './models/model.h5'
model_weights_path = './models/weights.h5'

parser = argparse.ArgumentParser(description='This program helps us to estimate the quality of learning.\nIt needs a trained InceptionResNetV2 network model, a path to directory containing dogs pictures to be tested and a correct category (class) number for these pictures.\nCorrect categories are:\n\n0 - Alaskan Malamute,\n1 - Pitbull,\n2 - Golden Retriever.\n\nThis software was written for BWI50202 .NET-Vertiefung Modul, Gruppe2,\nHochschule Niederrhein by Andrej Ovchinnikov, Jan Haupts and Yassine Magri.\nDecember 2018',
                                add_help=True,
                                epilog='Example of use: "python3 predict-test-dir-IResNetV2.py ./data/validation/Pitbull 1"',
                                formatter_class=RawTextHelpFormatter)
parser.add_argument("path", action ="store", help='Path to directory containing pictures')
parser.add_argument("category", action ="store", help='Correct class number for these pictures', type=int)
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()

if args.verbose:
    print("Verbosity turned on")

print("Das neuronale Netz und die Gewichte werden geladen. Bitte warten...")
model = load_model(model_path)
model.load_weights(model_weights_path) 
print("Das neuronale Netz und die Gewichte wurden erfolgreich geladen :)")

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  return answer

for i, ret in enumerate(os.walk(args.path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    elif filename.endswith(".jpg"):
        result = predict(ret[0] + '/' + filename)
        if result == int(args.category):
            print("Korrekte Erkennung (als", breeds[result], "): ", ret[0] + '/' + filename)
            counter_ok += 1
            continue
        else:
            print("Fehler: ", ret[0] + '/' + filename, end='')
            print(' - Erkannt als ', breeds[result])
            counter_wrong += 1

print("--------------------------------------------------------------------------------")
print("Gesamtergebnis:")
print("Korrekt erkannt:", counter_ok)
print("Erkannt mit Fehler:", counter_wrong)
print("Genauigkeit der Erkennung: %2.2f%%." % ((counter_ok/(counter_ok + counter_wrong))*100))

