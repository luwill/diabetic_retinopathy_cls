#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 13:37
# @Author  : louwill
# @File    : resnet50.py
# @mail: ygnjd2016@gmail.com


import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from itertools import cycle
from sklearn.utils import class_weight
from skll.metrics import kappa
import matplotlib.pyplot as plt

np.random.seed(1337)


class EyeNet:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_data_size = None
        self.weights = None
        self.model = None
        self.nb_classes = None
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.n_gpus = 2

    def split_data(self, y_file_path, X, test_data_size=0.2):
        """
        Split data into test and training data sets.

        INPUT
            y_file_path: path to CSV containing labels
            X: NumPy array of arrays
            test_data_size: size of test/train split. Value from 0 to 1

        OUTPUT
            Four arrays: X_train, X_test, y_train, and y_test
        """
        # labels = pd.read_csv(y_file_path, nrows=60)
        labels = pd.read_csv(y_file_path)
        self.X = np.load(X)
        self.y = np.array(labels['level'])
	# self.y = np.array([1 if l >=1 else 0 for l in labels['level']])
        self.weights = class_weight.compute_class_weight('balanced', np.unique(self.y), self.y)
        self.test_data_size = test_data_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_data_size,
                                                                                random_state=42)

    def reshape_data(self, img_rows, img_cols, channels, nb_classes):
        """
        Reshapes arrays into format for MXNet

        INPUT
            img_rows: Array (image) height
            img_cols: Array (image) width
            channels: Specify if image is grayscale(1) or RGB (3)
            nb_classes: number of image classes/ categories

        OUTPUT
            Reshaped array of NumPy arrays
        """
        self.nb_classes = nb_classes
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, channels)
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255

        self.y_train = np_utils.to_categorical(self.y_train, self.nb_classes)

        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, channels)
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255

        self.y_test = np_utils.to_categorical(self.y_test, self.nb_classes)

        print("X_train Shape: ", self.X_train.shape)
        print("X_test Shape: ", self.X_test.shape)
        print("y_train Shape: ", self.y_train.shape)
        print("y_test Shape: ", self.y_test.shape)


    def ResNet50_model(self, batch_size, nb_epoch):

        base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                              input_shape=(self.img_rows, self.img_cols, self.channels),
                              classes=self.nb_classes)

        for layer in base_model.layers:
            layer.trainable = False

        self.model = Flatten()(base_model.output)
        self.model = Dense(self.nb_classes, activation='softmax')(self.model)
        self.model = Model(inputs=base_model.input, outputs=self.model)

        sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
                       epochs=nb_epoch,
                       verbose=1,
                       validation_split=0.2,
                       class_weight=self.weights)

       # plot_model(self.model, to_file='resnet50_model.png', show_shapes=True)

        return self.model

    def predict(self):
        """
        Predicts the model output, and computes precision, recall, and F1 score.

        INPUT
            model: Model trained in Keras

        OUTPUT
            Precision, Recall, and F1 score
        """
        predictions = self.model.predict(self.X_test)
        predictions = np.argmax(predictions, axis=1)

       # predictions[predictions >=1] = 1 # Remove when non binary classifier

        self.y_test = np.argmax(self.y_test, axis=1)

        precision = precision_score(self.y_test, predictions, average='micro')
        recall = recall_score(self.y_test, predictions, average='micro')
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='micro')
        cohen_kappa = cohen_kappa_score(self.y_test, predictions)
        quad_kappa = kappa(self.y_test, predictions, weights='quadratic')
        report_result = classification_report(self.y_test, predictions)

        return accuracy, precision, recall, f1, cohen_kappa, quad_kappa, report_result

    def save_model(self, model_name):
	self.model.save("./model/" + model_name + ".h5")

if __name__ == '__main__':
    cnn = EyeNet()
    cnn.split_data(y_file_path="../data/labels/trainLabels_master_256_v2.csv", X="../data/X_train.npy")
    cnn.reshape_data(img_rows=256, img_cols=256, channels=3, nb_classes=5)
    model = cnn.ResNet50_model(256, 8)
    accuracy, precision, recall, f1, cohen_kappa, quad_kappa, report_result = cnn.predict()
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Cohen Kappa Score", cohen_kappa)
    print("Quadratic Kappa: ", quad_kappa)
    print("classification report: ", report_result)
    print(model.summary())
    cnn.save_model(model_name="DR_five_class")

