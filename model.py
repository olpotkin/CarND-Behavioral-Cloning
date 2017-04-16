import os
import csv


# Read driving data
# Catalogs with driving data:
drive_data_02 = 'drive-data-2'
drive_data_03 = 'drive-data-3'
# Log:
driving_log = '/driving_log.csv'

datasets = [drive_data_02, drive_data_03]

print('Loading datasets...')
samples = []
for dataset in datasets:
    ds = []
    with open(dataset + driving_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            ds.append(line)
        print('Dataset: {0} ({1} records)'.format(dataset, len(ds)))
        samples.extend(ds)
print('Loading completed ({0} records from {1} datasets)'.format(len(samples), len(datasets)))


# Split data
from sklearn.model_selection import train_test_split
# Split off 20% of the data to use for a test set.
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
print('Training records (no augmentation): {0}'.format(len(train_samples)))
print('Validation records (no augmentation): {0}'.format(len(validation_samples)))


import cv2
import numpy as np
import sklearn
import matplotlib
# Force matplotlib to not use any Xwindows backend (for AWS)
matplotlib.use('agg')
import matplotlib.pyplot as plt
#%matplotlib inline


# Update path to an image
def update_img_path(path):
    upd_path = path.split('/')[-3] + '/' + \
               path.split('/')[-2] + '/' + \
               path.split('/')[-1]
    return upd_path


# Crop the image (for plotting)
def img_crop(image):
    height = image.shape[0]
    width = image.shape[1]
    # Crop 70px from the Top
    # Crop 25px from the Bottom
    image = image[70:height-25, 0:width]
    return image


# Data augmentation - add flipped images and inverted angles to dataset
def flipping_augmentation(images, angles):
    aug_images, aug_angles = [], []
    for image, angle in zip(images, angles):
        aug_images.append(image)
        aug_angles.append(angle)
        # Flipping Images (horizontally) and inverting Steering Angles
        aug_images.append(np.fliplr(image))
        aug_angles.append(angle * -1.0)
    return aug_images, aug_angles


# Plot row (1x3) of images with titles
def plot_row_3(pics, titles):
    plt.figure(figsize=(12, 6))
    for i in range(0, 3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i], wrap=True)
        plt.imshow(pics[i])
    plt.tight_layout()
    plt.show()


# Plot images for report
def plot_imgs(img_center, img_left, img_right, angles):
    pics = [img_left, img_center, img_right]
    titles = ['Left camera (ang = {0:.2f})'.format(angles[1]),
              'Center camera (ang = {0:.2f})'.format(angles[0]), 
              'Right camera (ang = {0:.2f})'.format(angles[2])]
    plot_row_3(pics, titles)
    pics = [np.fliplr(img_left), np.fliplr(img_center), np.fliplr(img_right)]
    titles = ['Left camera (flip, ang = {0:.2f})'.format(angles[1] * -1.0), 
              'Center camera (flip, ang = {0:.2f})'.format(angles[0] * -1.0), 
              'Right camera (flip, ang = {0:.2f})'.format(angles[2] * -1.0)]
    plot_row_3(pics, titles)
    pics = [img_crop(img_left), img_crop(img_center), img_crop(img_right)]
    titles = ['Left camera (crop)', 'Center camera (crop)', 'Right camera (crop)']
    plot_row_3(pics, titles)
    pics = [np.fliplr(img_crop(img_left)), np.fliplr(img_crop(img_center)), np.fliplr(img_crop(img_right))]
    titles = ['Left camera (flip, crop)', 'Center camera (flip, crop)', 'Right camera (flip, crop)']
    plot_row_3(pics, titles)


def generator(samples, batch_size = 32):
    num_samples = len(samples)
    
    # Loop forever so the generator never terminates (set TRUE to plot images):
    show_plot = False
    
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
                correction = 0.2                         # parameter to tune
                angle_left = angle_center + correction
                angle_right = angle_center - correction

                # Read in images from center, left and right cameras
                path_center, path_left, path_right = batch_sample[0], batch_sample[1], batch_sample[2]
                
                # Update path to an images
                path_center = update_img_path(path_center)
                path_left   = update_img_path(path_left)
                path_right  = update_img_path(path_right)
                
                # Use OpenCV to load an images
                img_center = cv2.cvtColor(cv2.imread(path_center), cv2.COLOR_BGR2RGB)
                img_left   = cv2.cvtColor(cv2.imread(path_left), cv2.COLOR_BGR2RGB)
                img_right  = cv2.cvtColor( cv2.imread(path_right), cv2.COLOR_BGR2RGB)
                
                # Plot example of input and augmented images
                if show_plot:
                    angles = [angle_center, angle_left, angle_right]
                    plot_imgs(img_center, img_left, img_right, angles)
                    show_plot = False

                # Add Images and Angles to dataset
                images.extend([img_center, img_left, img_right])
                angles.extend([angle_center, angle_left, angle_right])

            # Data augmentation (flipping Images (horizontally) and inverting Steering Angles)
            aug_images, aug_angles = flipping_augmentation(images, angles)

            # Convert Images and Steering measurements to NumPy arrays
            # (the format Keras requires)
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)

            yield sklearn.utils.shuffle(X_train, y_train)


# Build the basic neural network to verify that everything is working
# Flattened image connected to a single output node. This single output
# node will predict steering angle, which makes this a regression network.
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import TensorBoard

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


# CNN architecture from Nvidia
# https://github.com/0bserver07/Nvidia-Autopilot-Keras
def nvidia_cnn():
    model = Sequential()
    # pixel_normalized = pixel / 255
    # pixel_mean_centered = pixel_normalized - 0.5
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation = "relu", subsample = (2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation = "relu", subsample = (2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation = "relu", subsample = (2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation = "relu", subsample = (1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation = "relu", subsample = (1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# Train model with the feature and label arrays.
model = nvidia_cnn()
model.summary()
model.compile(loss='mse', optimizer='adam')

# Callback to tensorboard to write logs
# Visualize model and its performance using tensorboard:
# --logdir = model_logs
# --write_graph = True (to visualize  the model)
tensorboard = TensorBoard(log_dir = "model_logs", write_graph = True, write_images = True)

history_object = model.fit_generator(
    train_generator, 
    samples_per_epoch = len(train_samples) * 6, 
    validation_data   = validation_generator,
    nb_val_samples    = len(validation_samples) * 6, 
    nb_epoch          = 12,
    verbose 		  = 1,
    callbacks		  = [tensorboard])

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_graph.png', bbox_inches='tight')

# Save the train model
model.save('model.h5')
exit()
