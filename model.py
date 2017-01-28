# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 20:53:55 2017

@author: setty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import json
import pickle
import os
import random

from keras.layers import Input,Dense, Dropout, Activation, Flatten,Lambda,ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K

from sklearn.utils import shuffle


### Load the training dataset from pickled file
file_Name = "training_dataset_resized.p"
with open(file_Name, mode='rb') as f:
    train_pickle = pickle.load(f)

### Center Camera
X_train_center,y_train_center = train_pickle['center']
### Left Camera
X_train_left,y_train_left = train_pickle['left']
### Right Camera
X_train_right,y_train_right = train_pickle['right']

#  Should pass these tests which verifies pickled data is loaded correctly. 
assert np.array_equal(X_train_center, train_pickle['center'][0]), 'X_train not set to data[\'center\'].'
assert np.array_equal(y_train_center,  train_pickle['center'][1]), 'y_train not set to data[\'center\'].'
assert np.array_equal(X_train_left, train_pickle['left'][0]), 'X_train not set to data[\'left\'].'
assert np.array_equal(y_train_left,  train_pickle['left'][1]), 'y_train not set to data[\'left\'].'
assert np.array_equal(X_train_right, train_pickle['right'][0]), 'X_train not set to data[\'right\'].'
assert np.array_equal(y_train_right,  train_pickle['right'][1]), 'y_train not set to data[\'right\'].'
print('Tests passed.')

X_train_center = np.array(X_train_center)
y_train_center = np.array(y_train_center)
X_train_left = np.array(X_train_left)
y_train_left = np.array(y_train_left)
X_train_right = np.array(X_train_right)
y_train_right = np.array(y_train_right)


### Recovery data
def recovery_data(camera,offset,Y):
    if camera == 'left':
        Y +=offset
    elif camera == 'right':
        Y -=offset
    return Y
    
y_train_left = recovery_data('left',0.25,y_train_left)
y_train_right = recovery_data('right',0.25,y_train_right)



###Flip for data augmentation( center camera images and corresponding -ve of steering angles)
### flipped images are appended
X_train_aug_flip =[]
l_center = len(X_train_center)

for i in range(l_center):
    img2=cv2.flip(X_train_center[i],1)
    X_train_aug_flip.append(img2)

### Augment the flipped images to unflipped images
X_train_center = np.append(X_train_center,X_train_aug_flip,axis=0)
### Modify the corresponding steering angles with (-ve) of that
y_train_center = np.append(y_train_center,-y_train_center,axis=0)

### Augment the Left and right camera data
X_train_aug = np.append(X_train_center,np.append(X_train_left,X_train_right,axis=0),axis=0)
y_train_aug = np.append(y_train_center,np.append(y_train_left,y_train_right,axis=0),axis=0)


### Brightness  augmentation for all images
X_train_aug_bright=[]

### Brightness routine adapted from Vivek Yadav's blog
### https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image.astype("uint8"),cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

for i in range(len(X_train_aug)):
    X_train_aug_bright.append(augment_brightness_camera_images(X_train_aug[i]))


X_train_aug = np.append(X_train_aug,X_train_aug_bright,axis=0)
y_train_aug = np.append(y_train_aug,y_train_aug,axis=0)

### Convert RGB-----> HSV and only 'S' component is used
for i in range(len(X_train_aug)):
    X_train_aug[i] = cv2.cvtColor(X_train_aug[i].astype("uint8"), cv2.COLOR_RGB2HSV)

### Shuffle the training dataset    
X_train,y_train = shuffle(X_train_aug[:,:,:,1],y_train_aug)

### Deep Neural Network model
ch, row, col = 1, 16, 32  # camera format
X_train = X_train.reshape(X_train.shape[0], row, col, ch)

model = Sequential() 

model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=( row, col,ch),output_shape=(row, col,ch)))
model.add(Convolution2D(12, 3, 3, border_mode="valid"))
model.add(ELU())
model.add(MaxPooling2D((4,4),(4,4),'valid'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=10, validation_split=0.2,shuffle=True)
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")