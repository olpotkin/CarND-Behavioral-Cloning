# Step 1 (Hello world):
# - Take in an image from the center camera of the car. This is the input to neural network.
# - Output a new steering angle for the car.
# Step 2 (Preprocessing):
# - Normalizing the data (Add a Lambda layer to the model)
# - Mean centering the data
# Step 3 (Data augmentation):
# Problem: The car seems to pull too hard to the left
# - Flipping Images (horizontally) and Invert Steering Angles
# Step 4 (Using multiple cameras)
#

import csv
import cv2
import numpy as np


# Read driving_log.csv file
lines = []
with open('drive-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read images and measurements
images = []
measurements = []
for line in lines:
    # Update path to an image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'drive-data/IMG/' + filename
    # Use OpenCV to load the image
    image = cv2.imread(current_path)
    images.append(image)
    # Extract the fourth token (Steering Angle) from the CSV line
    measurement = float(line[3])
    measurements.append(measurement)

# Data augmentation
aug_images, aug_measurements = [], []
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    # Flipping Images (horizontally) and invert Steering Angles
    aug_images.append(cv2.flip(image, 1))
    aug_measurements.append(measurement * -1.0)

# Convert Images and Steering measurements to NumPy arrays
# (the format Keras requires)
X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

# Build the basic neural network to verify that everything is working
# Flattened image connected to a single output node. This single output
# node will predict steering angle, which makes this a regression network.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# pixel_normalized = pixel / 255
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Train model with the feature and label arrays.
# Shuffle the data and split off 20% of the data to use for a validation set.
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train,
    batch_size = 128,
	validation_split=0.2,
	shuffle=True,
	nb_epoch=5)

# Save the train model
model.save('model.h5')
exit()

