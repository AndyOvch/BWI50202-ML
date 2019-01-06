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

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
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
  epochs = 2
else:
  epochs = 20

train_data_path = './data/train'
validation_data_path = './data/validation'

"""
Parameters
"""
img_width, img_height = 299, 299
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

base_model = InceptionResNetV2(include_top=False, input_shape=(299,299,3), weights='imagenet')

for layer in base_model.layers:
        layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
x = Dense(classes_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.Adam(lr=1e-3),
              metrics = ['acc'])

train_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	channel_shift_range=10,
	horizontal_flip=True,
	fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
	train_data_path,
	target_size=(img_height, img_width),
	interpolation='bicubic',
	class_mode='categorical',
	shuffle=True,
	batch_size=batch_size)

validation_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
	validation_data_path,
	target_size=(img_height, img_width),
	interpolation='bicubic',
	class_mode='categorical',
	shuffle=False,
	batch_size=batch_size)

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
