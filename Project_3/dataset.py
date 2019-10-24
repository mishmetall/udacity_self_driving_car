import pickle

import pickle
import numpy as np
from sklearn.utils import shuffle
import cv2
import skimage.morphology as morp
from skimage.filters import rank

class ds():
    def __init__(self, im, lb):
        self.images = im
        self.labels = lb
        self.classes = np.unique(lb).shape[0]
        self.shape = self.images[0]

    def __len__(self):
        return len(self.images)

    def shuffle(self):
        self.images, self.labels = shuffle(self.images, self.labels, random_state=0)

    def get_batch(self, start, end):
        labls = np.zeros((end-start+1, 43))
        labls[[i for i in range(end-start+1)], np.asarray(self.labels[start:end+1], dtype='int')] = 1.
        return self.images[start:end+1], labls


def get_data():

    training_file = "./traffic-signs-data/train.p"
    validation_file = "./traffic-signs-data/valid.p"
    testing_file = "./traffic-signs-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    assert len(X_train) == len(y_train)
    X_valid, y_valid = valid['features'], valid['labels']
    assert len(X_valid) == len(y_valid)
    X_test, y_test = test['features'], test['labels']




    def gray_scale(image):
        """
        Convert images to gray scale.
            Parameters:
                image: An np.array compatible with plt.imshow.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def local_histo_equalize(image):
        """
        Apply local histogram equalization to grayscale images.
            Parameters:
                image: A grayscale image.
        """
        kernel = morp.disk(30)
        img_local = rank.equalize(image, selem=kernel)
        return img_local
    #
    # def image_normalize(image):
    #     """
    #     Normalize images to [0, 1] scale.
    #         Parameters:
    #             image: An np.array compatible with plt.imshow.
    #     """
    #     image = np.divide(image, 255)
    #     return image
    #
    # def preprocess(data):
    #     """
    #     Applying the preprocessing steps to the input data.
    #         Parameters:
    #             data: An np.array compatible with plt.imshow.
    #     """
    #     gray_images = list(map(gray_scale, data))
    #     equalized_images = list(map(local_histo_equalize, gray_images))
    #     n_training = data.shape
    #     normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    #     for i, img in enumerate(equalized_images):
    #         normalized_images[i] = image_normalize(img)
    #     normalized_images = normalized_images[..., None]
    #     return normalized_images
    #
    # X_train = preprocess(X_train)
    # X_valid = preprocess(X_valid)
    # X_test = preprocess(X_test)

    return ds(np.asarray(X_train, dtype=np.float) / 255., np.asarray(y_train, dtype=np.float)), \
           ds(np.asarray(X_valid, dtype=np.float) / 255., np.asarray(y_valid, dtype=np.float))



# # TODO: Number of training examples
# n_train = X_train.shape[0]
#
# import cv2
#
# for i in range(len(X_valid)):
#     cv2.imshow("sdc", X_valid[i])
#     cv2.waitKey(0)
# # TODO: Number of validation examples
# n_validation = X_valid.shape[0]
#
# # TODO: Number of testing examples.
# n_test = X_test.shape[0]
#
# # TODO: What's the shape of an traffic sign image?
# image_shape = X_train.shape[1:3]
#
# # TODO: How many unique classes/labels there are in the dataset.
# n_classes = np.unique(y_train).shape[0]
#
# print("Number of training examples =", n_train)
# print("Number of testing examples =", n_test)
# print("Image data shape =", image_shape)
# print("Number of classes =", n_classes)