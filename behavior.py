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
# Step 5 (Using Generators to prevent MemoryError)
#

import os
import csv


# Read driving_log.csv file
samples = []
with open('drive-data-2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data
from sklearn.model_selection import train_test_split
# Split off 20% of the data to use for a test set.
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)


import cv2
import numpy as np
import sklearn


def generator(samples, batch_size = 32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates:
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Extract the fourth token (Steering Angle)
                angle_center = float(batch_sample[3])

                # Create adjusted steering measurements for the side camera images
                # Parameter to tune:
                correction = 0.2
                angle_left = angle_center + correction
                angle_right = angle_center - correction

                # Read in images from center, left and right cameras
                path_center, path_left, path_right = batch_sample[0], batch_sample[1], batch_sample[2]
                # Update path to an images
                path_center = 'drive-data-2/IMG/' + path_center.split('/')[-1]
                path_left   = 'drive-data-2/IMG/' + path_left.split('/')[-1]
                path_right  = 'drive-data-2/IMG/' + path_right.split('/')[-1]

                # Use OpenCV to load an images
                img_center = cv2.imread(path_center)
                img_left   = cv2.imread(path_left)
                img_right  = cv2.imread(path_right)

                # Add Images and Angles to dataset
                images.extend([img_center, img_left, img_right])
                angles.extend([angle_center, angle_left, angle_right])
                #images.append(img_left)
                #angles.append(angle_left)
                #images.append(img_right)
                #angles.append(angle_right)

            # Data augmentation
            aug_images, aug_angles = [], []
            for image, angle in zip(images, angles):
                aug_images.append(image)
                aug_angles.append(angle)
                # Flipping Images (horizontally) and invert Steering Angles
                #aug_images.append(cv2.flip(image, 1))
                aug_images.append(np.fliplr(image))
                aug_angles.append(angle * -1.0)

            # TODO: Trim image to only see section with road
            # Convert Images and Steering measurements to NumPy arrays
            # (the format Keras requires)
            
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            #X_train = np.array(images)
            #y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Build the basic neural network to verify that everything is working
# Flattened image connected to a single output node. This single output
# node will predict steering angle, which makes this a regression network.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)

model = Sequential()
# pixel_normalized = pixel / 255
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation = "relu"))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train model with the feature and label arrays.
model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    train_generator, 
    samples_per_epoch = len(train_samples) * 6, 
    validation_data   = validation_generator,
    nb_val_samples    = len(validation_samples) * 6, 
    nb_epoch          = 3)

#model.fit(X_train, y_train,
#    batch_size = 128,
#	validation_split = 0.2,
#	shuffle = True,
#	nb_epoch = 5)

# Save the train model
model.save('model.h5')
exit()
