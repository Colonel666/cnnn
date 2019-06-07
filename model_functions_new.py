# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:37:40 2019

@author: Wolfgang Reuter
"""

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, AveragePooling2D
from keras import initializers

def set_up_top_res50(num_channels, dropout_rate):
    """
    Sets up a custom top model for the output of a Resnet50
    """
    a = Input(shape=(7,7,2048))
    x = AveragePooling2D((7, 7), name='avg_pool')(a)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(2048, \
              kernel_initializer=initializers.he_normal(), \
              bias_initializer=initializers.ones(), \
              activation='relu')(x)
    x = Dense(num_channels, kernel_initializer=initializers.he_normal(), \
              bias_initializer=initializers.ones(), activation='softmax')(x)
    model = Model(inputs=a, outputs=x)
    
    return model


def cust_res50_classifier(n_classes, dropout_rate, top_model_weights_path=''):
    
    from keras.applications.resnet50 import ResNet50
    
    in_shape = (224,224,3)
    
    # Get top model architecture
    top_model = set_up_top_res50(n_classes, dropout_rate)
    
    # Get top model weights
    if top_model_weights_path != '':
        top_model.load_weights(top_model_weights_path)
    
    top_model.summary()
    
    # Merge the two models using Functional API
    inp = Input(shape=in_shape)
    y = ResNet50(include_top=False, \
                 weights='imagenet', \
                 input_shape=in_shape)(inp)
    
    y = top_model(y)
    
    final_model = Model(inputs=inp, output = y)
    
    return final_model
