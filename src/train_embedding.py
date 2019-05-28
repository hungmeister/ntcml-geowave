import torch
import torch.nn as nn
import pickle as cPickle
import math
import os
import Embedder
import Classifier
import VehicleDataset
import torch.optim as optim
import eval
import shutil
import state_utils
from config import USE_GPU, EMBEDDING_CHECKPOINT, SOFTMAX_CHECKPOINT_EMBED, DATASET, \
    EMBEDDING_BATCH_SIZE, EMBEDDING_EPOCHS, ALPHA, EMBEDDING_LR, EMBEDDING_MOMENTUM, \
    EMBEDDING_WEIGHT_DECAY

#from tensorboard_logger import configure, log_value
dataset = os.path.join(DATASET, "embedding")

#configure("/home/mgregory/wami/wami-linker-new/tensorboard_out", flush_secs=5)

####################################### a few functions##########################################################################

# Circularly iterate through arrays
def wrap_index(length, index):
    if index >= length:
        index = index - length

    return index

############################ triplet loss class ################################################################################
# Triplet loss function for embedder training
class triplet_loss(torch.nn.Module):

    def __init__(self):
        super(triplet_loss, self).__init__()
        # Initialize threshold

    def forward(self, outputs, labels):
        losses = []

        # Sort by label to form class groups
        sorted_by_label = zip(outputs, labels)
        sorted_by_label = sorted(sorted_by_label, key=lambda x: x[1])
        n = len(sorted_by_label)

        group_indeces = [-1 for i in sorted_by_label]  # Indices corresponding to group changes
        group_indeces[0] = sorted_by_label[0][1]
        first = True
        for label in group_indeces:
            # Do not repeat loop on same group
            if label == -1:
                continue
            start_index = group_indeces.index(label)  # (Where to stop loop)
            p1 = sorted_by_label[start_index][0]  # First positive
            j = 1  # Index for second positive
            # Find second positive while within same group as first positive
            while group_indeces[wrap_index(n, start_index + j)] == -1:
                i = wrap_index(n, start_index + j)
                stop_after_search = False
                if sorted_by_label[i][1] == label:
                    p2 = sorted_by_label[i][0]
                    ap = torch.dot(p1 - p2, p1 - p2).float()
                else:
                    stop_after_search = True
                if not stop_after_search:
                    prev_label = sorted_by_label[i][1]
                    i = wrap_index(n, i + 1)
                else:
                    prev_label = sorted_by_label[i - 1][1]
                first_group = True
                while i != start_index:
                    cur_entry = sorted_by_label[i]
                    cur_label = cur_entry[1]
                    if cur_label != prev_label:
                        first_group = False
                        if first:
                            group_indeces[i] = cur_label
                        prev_label = cur_label
                    if not stop_after_search and not first_group:
                        # set negative
                        negative = cur_entry[0]
                        # Calculate positive-negative L2 distances
                        an1 = torch.dot(p1 - negative, p1 - negative).float()
                        an2 = torch.dot(p2 - negative, p2 - negative).float()
                        if USE_GPU:
                            loss = torch.tensor(0).float().cuda()
                        else:
                            loss = torch.tensor(0).float()
                        if ap < an1 and ap - an1 + ALPHA > 0:
                            loss += (ap - an1 + ALPHA).float()
                        if ap < an2 and ap - an2 + ALPHA > 0:
                            loss += (ap - an2 + ALPHA).float()
                        losses.append(loss)
                    i = wrap_index(n, i + 1)
                first = False
                j += 1

        # I was getting a strange memory issue when I tried adding up all the losses at once,
        # so I added them up in batches and it solved the problem
        bucket_size = 100
        if USE_GPU:
            total_loss = torch.tensor(0).float().cuda()
        else:
            total_loss = torch.tensor(0).float()
        for i in range(int(math.ceil(float(len(losses))/bucket_size))):
            start = i*bucket_size
            if USE_GPU:
                bucket_loss = torch.tensor(0).float().cuda()
            else:
                bucket_loss = torch.tensor(0).float()
            for j in range(bucket_size):
                index = start + j
                if index >= len(losses):
                    break
                bucket_loss += losses[index]
            total_loss += bucket_loss
        return total_loss


########################### train method acting like main method #############################
def train(embedder, optimizer, criterion, trainset, testset, checkpoint_dir, logs_dir):
    print("in train_embedding train method")
    embedder.train(True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=EMBEDDING_BATCH_SIZE,
                                              shuffle=True, num_workers=1)

    for epoch in range(EMBEDDING_EPOCHS):  # loop over the dataset multiple times
        losses = []

        print("Epoch: " + str(epoch))
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if list(labels.size())[0] != EMBEDDING_BATCH_SIZE:
                continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if USE_GPU:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = embedder(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            losses.append(loss.item())
            if i % 25 == 0:
                running_loss = 0
                print("Loss: " + str(loss.item()))
                #log_value('loss', loss.item(), epoch + 1)
            if loss != 0:
                loss.backward()
                optimizer.step()
        eval.log_embedding(embedder, trainset, testset, losses, os.path.join(logs_dir, "train_log"))
        state_utils.save_state(checkpoint_dir, embedder, optimizer, str(epoch))


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
    classifier = Classifier.Net()
    classifier = nn.DataParallel(classifier)
    classifier = state_utils.load_state(classifier, SOFTMAX_CHECKPOINT_EMBED)
    embedder = Embedder.Net(classifier)
    embedder = nn.DataParallel(embedder)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, embedder.parameters()), lr=EMBEDDING_LR, momentum=EMBEDDING_MOMENTUM,
                          weight_decay=EMBEDDING_WEIGHT_DECAY)  # lr = .001

    # Load checkpoint if provided
    if EMBEDDING_CHECKPOINT != "":
        embedder, optimizer = state_utils.load_state(embedder, EMBEDDING_CHECKPOINT, optimizer=optimizer)

    criterion = triplet_loss()

    if USE_GPU:
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            embedder = embedder.cuda()

    # Initialize datsets
    trainset = VehicleDataset.VehicleDataset(os.path.join(dataset, 'train.csv'))
    testset = VehicleDataset.VehicleDataset(os.path.join(dataset, 'test.csv'))

    ## Train
    train(embedder, optimizer, criterion, trainset, testset, checkpoint_dir, logs_dir)



