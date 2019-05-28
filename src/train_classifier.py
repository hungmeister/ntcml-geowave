import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import VehicleDataset
import Classifier
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import cv2
import pickle as cPickle
import shutil
import os
#from tensorboard_logger import configure, log_value
import eval
import state_utils
from config import USE_GPU, SOFTMAX_LR, SOFTMAX_BATCH_SIZE, SOFTMAX_EPOCHS, DATASET, \
    SOFTMAX_CHECKPOINT, SOFTMAX_MOMENTUM, SOFTMAX_WEIGHT_DECAY
import platform
print(platform.python_version())

print(SOFTMAX_CHECKPOINT)

#configure("../tensorboard_out", flush_secs=5)
dataset_dir = os.path.join(DATASET, "cnn")

######################### CLASSSES AND FUNCTIONS #########################
# denormalize and display image
def imshow(img, mean, std):
    img = (img*std + mean)*255.0
    img = np.asarray(img.numpy())
    img = img[0][0]
    img = Image.fromarray(img)
    img.show()

# Train classifier
def train(net, optimizer, criterion, trainset, testset, checkpoint_dir, logs_dir):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=SOFTMAX_BATCH_SIZE,
                                              shuffle=True, num_workers=1)

    for epoch in range(SOFTMAX_EPOCHS):
        # Set model to train mode
        net.train(True)
        # Log loss at each iteration within each epoch
        losses = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # Skip batch if remainder is smaller than set batch size
            if list(labels.size())[0] != SOFTMAX_BATCH_SIZE:
                continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save loss
            loss = loss.item()
            losses.append(loss)

            # print statistics
            if i % 10 == 9:    # print every n mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
                #log_value('loss', loss, epoch + 1)

        print("iteration complete. Moving to eval")
        # Save model state as checkpoint
        state_utils.save_state(checkpoint_dir, net, optimizer, str(epoch))
        # Log statistics
        eval.log_classifier(net, trainset, testset, losses, os.path.join(logs_dir, "train_log"))


########## MAIN ###################

if __name__ == "__main__":

    ## Handle file structure
    parent_dir = "../"
    folders = os.listdir(parent_dir)
    logs_dir = os.path.join(parent_dir, "logs")
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    if "logs" in folders:
        shutil.rmtree(logs_dir)
    if "checkpoints" in folders:
        shutil.rmtree(checkpoint_dir)
    os.mkdir(logs_dir)
    os.mkdir(checkpoint_dir)

    ## Initialize
    net = Classifier.Net()
    # hand-off to GPU
    if USE_GPU:
        if torch.cuda.device_count() > 0:
            net = net.cuda()
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    # wrapper for parallel processing
    net = nn.DataParallel(net)
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    # Initialize Optimizer
    optimizer = optim.SGD(net.parameters(), lr=SOFTMAX_LR, momentum=SOFTMAX_MOMENTUM, weight_decay=.0005)  # lr = .001

    # Load checkpoint if provided
    if SOFTMAX_CHECKPOINT != "":
        net, optimizer = state_utils.load_state(net, SOFTMAX_CHECKPOINT, optimizer=optimizer)

    # Initialize datasets
    trainset = VehicleDataset.VehicleDataset(os.path.join(dataset_dir, 'train.csv'))

    testset = VehicleDataset.VehicleDataset(os.path.join(dataset_dir, 'test.csv'), flip=False)

    ## Train
    train(net, optimizer, criterion, trainset, testset, checkpoint_dir, logs_dir)
    print("Out of the train method")

