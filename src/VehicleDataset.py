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

class VehicleDataset(Dataset):

    def __init__(self, csv_file, transform=None, flip=True):
        print("inside Vehicle Dataset init function")
        dataset_dir = os.path.dirname(csv_file)
        # Parse csv that contains labels and points to data
        data = open(csv_file).readlines()[1:]
        random.shuffle(data)
        data = [entry.split(',') for entry in data]
        for entry in data:
            entry[1] = entry[1].split('\n')[0]
        self.labels = [entry[1] for entry in data]
        # Open data as numpy arrays and normalize
        data = [np.array(Image.open(os.path.join(dataset_dir, entry[0])))/255.0 for entry in data]
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.data = (data - self.mean)/self.std
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
        # Reshape data into pytorch batch configuration
        image, label = [torch.from_numpy(np.rot90(self.data[index], rot_90).copy().reshape(1, IMAGE_SIZE, IMAGE_SIZE)).float(), int(self.labels[index])]

        if self.transform:
            image = self.transform(image)

        return [image, label]



