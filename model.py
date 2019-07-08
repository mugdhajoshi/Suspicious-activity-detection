import numpy as np
#import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path='train'
valid_path='valid'
test_path='test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['suspicious','notsuspicious'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['suspicious','notsuspicious'], batch_size=4)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['suspicious','notsuspicious'], batch_size=10)

img_width=224
img_height=224

model = Sequential()
model.add(Conv2D(4,(3,3),strides=1, padding='valid', activation = 'relu', input_shape = (img_width,img_height,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))

model.add(Conv2D(16,(3,3), activation = 'relu', strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))

model.add(Conv2D(32, (3,3), activation="relu", strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(64, (3,3), activation="relu", strides=1, padding="valid"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(2, activation='sigmoid'))


model.compile(Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model_fit=model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=15,verbose=2)


predictions = model.predict_generator(test_batches,steps=1, verbose=0)
print (predictions)

#save the model 
#serialize model to JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights('model.h5')

