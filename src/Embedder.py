import torch
import torch.nn as nn
import numpy as np
import Classifier
from config import EMBEDDING_SIZE

final_feature_dim = 64

# Vector normalization module
class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        for i in range(len(x)):
            norm = torch.norm(x[i]).item()
            if norm == 0:
                x[i] = torch.tensor([0 for _ in x[i]]).float()
            else:
                x[i] = (x[i]/norm).float()

        return x

# Embedding Model
class Net(nn.Module):

    def __init__(self, trained_model):
        super(Net, self).__init__()
        # Extract layers from model
        model = list(list(trained_model.children())[0].children())
        # Remove FC block
        cnn = model[:-1]
        new_model = cnn
        # Combine layers list
        new_model = nn.Sequential(*new_model)
        # Freeze all layers before adding embedding layer
        for param in new_model.parameters():
            param.requires_grad = False
        # Add embedding layers and stitch everything together
        self.new_model = nn.Sequential(new_model, nn.Linear(final_feature_dim, final_feature_dim), nn.ReLU(), nn.Linear(final_feature_dim, EMBEDDING_SIZE))
        # Normalize outputs onto hypersphere
        self.norm = Normalize()

    def forward(self, x):
        x = self.new_model(x)
        x = self.norm(x)

        return x
