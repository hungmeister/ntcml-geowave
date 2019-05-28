from __future__ import print_function, division
import os
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from PIL import Image
import Classifier
import cv2
from config import IMAGE_SIZE
import glob

class MSTAR_7_Dataset(Dataset):

    def __init__(self, mstar7_dataset_dir, transform=None, flip=False):
        print("inside MSTAR_7_2S1 Dataset init function")
        mstar7_folders = os.listdir(mstar7_dataset_dir)
        # Traverse dataset_dir and extract data/images and labels
        # Extract images
        mstar7_imgs = []
        for folder in mstar7_folders:
            for img in (glob.glob(os.path.join(mstar7_dataset_dir, folder) + '/*.png')):
                mstar7_imgs.append(img)
        # Convert image to numpy array and normalize
        mstar7_data = [np.array(Image.open(img))/255.0 for img in mstar7_imgs]

        #random.shuffle(data)
        mstar7_labels = []
        for folder in os.listdir(mstar7_dataset_dir):
            for label in (glob.glob(os.path.join(mstar7_dataset_dir, mstar7_folders[0]) + '/*.png')):
                mstar7_labels.append(folder)

        # Convert labels from string or categorical to int
        #convert_labels_2_ints = {label: num for label,num in zip(mstar7_folders, \
        #                                list(range(1,len(mstar7_folders)+1)))}
        # From 0 to 6 for 7 labels, instead of 1 to 7
        convert_labels_2_ints = {label: num for label,num in zip(mstar7_folders, \
                                        list(range(0,len(mstar7_folders)+1)))}
        
        mstar7_labels_int = [convert_labels_2_ints[label] for label in mstar7_labels]
        # Pair all normalized arrays with their corresponding label
        mstar7_data_label_zipped = zip(mstar7_data, mstar7_labels_int)
        mstar7_data_n_labels = list(mstar7_data_label_zipped)

        # Shuffle
        random.shuffle(mstar7_data_n_labels)

        self.mean = np.mean(mstar7_data)
        self.std = np.std(mstar7_data)
        self.data = (mstar7_data - self.mean)/self.std
        self.data_n_label = mstar7_data_n_labels

        # Set transformation functions
        self.transform = transform
        self.flip = flip


    def get_mean_std(self):
        return [self.mean, self.std]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # If indicated, randomly rotate by some integer multiple of 90 degrees
        rot_90 = 0
        if self.flip:
            rot_90 = random.randint(0, 3)
        img_rotated = np.rot90(self.data_n_label[index][0], rot_90).copy()
        #img_one_channel = np.resize(img_rotated, (IMAGE_SIZE, IMAGE_SIZE, 1))
        img_one_channel = img_rotated[:,:,0]
        # Reshape data into pytorch batch configuration
        #image, label = [torch.from_numpy(np.rot90(self.data_n_label[index][0], rot_90).copy().reshape(1, IMAGE_SIZE, IMAGE_SIZE)).float(), self.data_n_label[index][1]]
        image, label = [torch.from_numpy(img_one_channel.reshape(1, IMAGE_SIZE, IMAGE_SIZE)).float(), self.data_n_label[index][1]]
        #image, label = [torch.from_numpy(np.rot90(self.data_n_label[index][0], rot_90).copy()).float(), self.data_n_label[index][1]]

        if self.transform:
            image = self.transform(image)

        return [image, label]



