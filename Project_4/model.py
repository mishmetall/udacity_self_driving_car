import cv2
import csv
from random import shuffle
import keras
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Conv2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from os import listdir
from os.path import join, basename
import sklearn
from sklearn.model_selection import train_test_split

# Path to dataset folder
dataset = 'data/'

samples = []
with open(join(dataset, "driving_log.csv")) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)


# Split dataset into train and validation one
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def read_im_rgb(im_path):
    return cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                central_im_name = join(dataset, "IMG", basename(batch_sample[0]))
                left_im_name = join(dataset, "IMG", basename(batch_sample[1]))
                right_im_name = join(dataset, "IMG", basename(batch_sample[2]))
                
                # measurement for central camera
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                central_im = read_im_rgb(central_im_name)
                left_im = read_im_rgb(left_im_name)
                right_im = read_im_rgb(right_im_name)
                
                images.extend([central_im, left_im, right_im])
                angles.extend([steering_center, steering_left, steering_right])

                # extend dataset with flipped ones
                images.extend([np.fliplr(central_im), np.fliplr(left_im), np.fliplr(right_im)])
                angles.extend([-steering_center, -steering_left, -steering_right])
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# set up lambda layer
model = Sequential()

#Preprocessing by scaling to -0.5...0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop unuseful data (ego-vehicle and sky)
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Rest of the model
model.add(Conv2D(24, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3, 3), strides=(2,2), activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

# Use early stopping instead of hardcoding epochs
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Save only best models
mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Train model
model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=100, verbose=1, callbacks=[es, mc])
