"""
Dieses Programm trainiert das neuronale Netz.
Dafür werden die Daten aus dem "dataset"-Verzeichnis verwendet.
Verwendung: 'python3 train-netzwerk.py'
(am besten zusamen mit 'nice' ausführen, da das Training lange 
dauert und sehr rechenintensiv ist)
"""

import sys
import os
import numpy as np


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import callbacks
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 3
else:
  epochs = 30

train_data_path = './data/train'
validation_data_path = './data/validation'

"""
Parameters
"""
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32
samples_per_epoch = 2000
validation_samples = 235
filters1 = 32
filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 3
lr = 0.0004

vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16_net.trainable = False

model = Sequential()

model.add(vgg16_net)

# Добавляем в модель новый классификатор
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes_num))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr=lr),
              metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch = samples_per_epoch // batch_size,
    epochs = epochs,
    verbose = 1, 
    workers = 1, 
    use_multiprocessing = False, 
    validation_data = validation_generator,
    callbacks = cbks,
    validation_steps = validation_samples // batch_size)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

model.save('./models/model.h5')
model.save_weights('./models/weights.h5')
