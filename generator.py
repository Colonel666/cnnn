# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:27:02 2019

@author: wolfg
"""

import numpy as np
import keras
from PIL import Image

from keras.applications.resnet50 import preprocess_input

class Img_DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras, labels are one hot, image data is preprocessed 
    according to keras procedure AND normalized by division through 255.0'
    """
    def __init__(self, list_IDs,labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            # Store sample
            #img = image.load_img(ID, target_size=(224, 224))
            x = Image.open(ID)
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X[i,] = x / 255.0
            
            # Store label vector
            # Store class as one hot vector with one entry for each class
            a = np.array(self.labels[ID]) - 1
            y[i] = keras.utils.to_categorical(a, num_classes=self.n_classes)
            
            return X, y
