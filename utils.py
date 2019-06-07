# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:50:26 2019

@author: wolfg
"""

import os
import numpy as np

def getAllConvNetImages(rootdir):
    """
    getAllConvNetImages(rootdir)
    
    returns the list of all images used in the painter classification project
    
    Arguments:
        
        rootdir         -   the path to the folder where the images for the 
                            ConvNet are stored, currently image_data in 
                            this project folder
        
    Output:
        
        pic_list        -   the list with all paths (strings) to the pictures
                            used in the painter classification project,
                            excluding Thumbs.db files
    """
    pic_list = []
    
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if 'Thumbs.db' not in file:
                pic_list.append(os.path.join(subdirs, file))
                
    return pic_list

def getPainters(pic_list):
    """
    getPainters(pic_list)
    
    returns the unique paninter names in a pic_list passed in as argument
    
    Arguments:
        
        pic_list        -   the list with the paths to the images for which
                            the dimensions are returned
        
    Output:
        
        np.unique(painter_list)    -   the numpy array with the unique 
                                       painter names in pic_list
    
    """
    
    painter_list = list()
    
    
    for i in range(0, len(pic_list)):
        
        im_arr = pic_list[i].split('/')
        
        painter = im_arr[-1].rsplit('_')
        
        painter_list.append(painter[0])
        
    return np.unique(painter_list)##

def setUpLabels(pic_list, painter_arr):
    """
    setUpLabels(pic_list, painter_arr)
    
    sets up a numpy array with the labels for each entry in pic_list.
    
    Arguments:
        
        pic_list        -   the list with all paths (strings) to the pictures
                            used in the painter classification project,
                            excluding Thumbs.db files
                            
        painter_arr     -   the numpy array with the unique 
                            painter names in pic_list
                            
    Output:
        
        y               -   the numpy array with the labels for each entry
                            in pic_list
                            
        NOTE: Primarily used to get the stratified split of pic_list for 
        setting up training, validation and test sets
        
        NOTE: Returns labels from 1 to number of painters in pic_list. If 
        labels are passed into keras to_categorical, NEED TO SUBTRACT 1 
        
    """
    
    m = len(pic_list)
    n = painter_arr.shape[0]
    
    y = np.zeros(m)
    
    for i in range(0, m):
        
        for j in range(0, n):
            
            if painter_arr[j] in pic_list[i]:
                
                y[i] = j + 1
                
    return y


def get_img_data(img_data_directory, set_name_list):
    """
    get_img_data(img_data_directory, img_data_directory, set_name_list)
    
    returns the lists with paths to the stored images for the 
    training, validation and test set as well as the labels dictionary 
    for all sets - ready to be used in an image data_generator
    
    Arguments: 
        
        img_data_directory          - The directory where the images are stored 
                                      (in train or train_aug_xxx, val and test
                                      subdirectories), i.e.: 
                                          'D:\\PC_ConvNet\\img_data\\train\\'
                                      
                                      
        set_name_list               - The list with the exact names of the 
                                      subdirectories required (i.e. 
                                      ['train_aug_1000', 'val', 'test']
                                      
    Output:
        
        train_data_list             - The list with the paths to the 
                                      training images
        
        val_data_list               - The list with the paths to the
                                      validation images
        
        test_data_list              - The list with the paths to the
                                      test images
        
        labels_dict                 - The dictionary with the paths to 
                                      all images (no matter in which set) as 
                                      key and the according label as value
    """
    
    train_arr_directory = img_data_directory + set_name_list[0]
    val_arr_directory = img_data_directory + set_name_list[1]
    test_arr_directory = img_data_directory + set_name_list[2]
    
    train_imgs_paths = getAllConvNetImages(train_arr_directory)
    val_imgs_paths = getAllConvNetImages(val_arr_directory)
    test_imgs_paths = getAllConvNetImages(test_arr_directory)
        
    painter_arr = getPainters(test_imgs_paths)
        
    train_labels_list = setUpLabels(train_imgs_paths, painter_arr)
    val_labels_list = setUpLabels(val_imgs_paths, painter_arr)
    test_labels_list = setUpLabels(test_imgs_paths, painter_arr)
    
    train_labels_dict = {}
    val_labels_dict = {}
    test_labels_dict = {}
    
    for i, path in enumerate(train_imgs_paths):
        train_labels_dict[path] = train_labels_list[i]
        
    for i, path in enumerate(val_imgs_paths):
        val_labels_dict[path] = val_labels_list[i]
        
    for i, path in enumerate(test_imgs_paths):
        test_labels_dict[path] = test_labels_list[i]
    
    labels_dict = train_labels_dict
    labels_dict.update(val_labels_dict)
    labels_dict.update(test_labels_dict)
    
    assert(len(labels_dict) == \
           len(train_imgs_paths) + len(val_imgs_paths) + len(test_imgs_paths))
    
    return train_imgs_paths, val_imgs_paths, test_imgs_paths, labels_dict