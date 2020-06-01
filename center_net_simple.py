from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print(tf.__version__) # tested with 1.15.0
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.models import Model
from keras import regularizers


## Loads all images in "image_directory" folder and runs the prediction network on them.
## Predicted centers are stored in "center_list" list and printed to console at program end.
## Prediction is x-coordinate between 0 and 150 regardless of input image size.
## Weights and test images can be found here:
## https://drive.google.com/open?id=1F-IUc-o722z8D4J30AM69owYwB5gWS2w

image_directory = "images" # directory with images to predict



img_size = 150

images = []
labels = []

IMG_WIDTH = img_size
IMG_HEIGHT = img_size
IMG_CHANNELS = 3

# MODEL
reg = tf.keras.regularizers.l2(0.0005)
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(inputs)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(p1)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(p2)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(p3)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(p4)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', kernel_regularizer=reg)(c5)

c6 = tf.keras.layers.Flatten()(c5)
c6 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=reg)(c6) #######
c6 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg)(c6)
c6 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg)(c6)
c6 = tf.keras.layers.Dense(img_size, activation='softmax')(c6)

outputs = c6

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Loads the weights
model.load_weights("weights/cp.ckpt")


images = []
centers = []

# LOAD IMAGES
directory = image_directory
print("Loading images..")
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(directory, filename))
        width = img.shape[1]
        resize_factor = width / img_size
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # cv2.COLOR_BGR2GRAY)  # BGR -> RGB
        img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
        images.append(img / 255.0)
images = np.asarray(images)
        
print("Predicting..")
predictions = model.predict(images)
for i in range(len(predictions)):
    pred_val = predictions[i].argsort()[-1:][::-1]
    centers.append(int(abs(pred_val)))

print(centers) # prints all predicted center locations (value between 0 to 150)

# displays image + prediction
for i in range(len(centers)):
    plt.imshow(images[i])
    plt.axvline(x=centers[i], color='green', label='Prediction')
    print("Image + prediction ", i)
    plt.show()

quit()

    


