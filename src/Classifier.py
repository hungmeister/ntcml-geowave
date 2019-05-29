import torch
import torch.nn as nn
import numpy as np
from config import IMAGE_SIZE, NUM_CLASSES, SOFTMAX_DROPOUT_PROB

# Flattening Module (for FC layers)
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Main CNN + FCs
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First convolutional block + max pooling
        self.conv_block1 = [nn.Conv2d(1, 192, 5, padding=2), nn.ReLU()]
        for i in range(5):
            self.conv_block1 = self.conv_block1 + [nn.Conv2d(192, 192, 5, padding=2), nn.ReLU()]
        self.conv_block1.append(nn.MaxPool2d(4, 4))
        self.conv_block1 = nn.Sequential(*iter(self.conv_block1))

        # Second convolutional block + max pooling + flattening
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 96, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(6, 6),
            Flatten()
        )

        # FC block containing dropout (to be removed for embedding)
        self.linear_block = nn.Sequential(
            nn.Dropout(SOFTMAX_DROPOUT_PROB),
            #nn.Linear(64, 77),
            nn.Linear(96, 77),
            nn.ReLU(),
            nn.Dropout(SOFTMAX_DROPOUT_PROB),
            nn.Linear(77, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.linear_block(x)

        return x
