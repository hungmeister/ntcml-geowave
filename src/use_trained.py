import Embedder
import Classifier
import eval
import state_utils
import torch.nn as nn
import torch
import VehicleDataset
import os

dataset_dir = "../cnn_data"
testset = VehicleDataset.VehicleDataset(os.path.join(dataset_dir, 'test.csv'), flip=False)

model = state_utils.load_state(nn.DataParallel(Classifier.Net()), "../experiments/cnn/new_code_50c/checkpoints/3094")
acc = eval.classification_acc(model, testset, n=100)
print(acc)