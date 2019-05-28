import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import VehicleDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pickle as cPickle
import train
import math
import random
import psutil
import timeit
import gc
import os
import shutil
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

use_gpu = True
transfer_learning = False
softmax_checkpoint = "/home/mgregory/wami/wami-linker-new/softmax_checkpoints/0"

lr = .0001  #.00025 for old stuff
batch_size = 50
embedding_size = 3
a = .2  # Triplet loss threshold
final_feature_dim = 40*1

loss_plot = []
acc_plot = []

# Circularly iterate through arrays
def wrap_index(length, index):
    if index >= length:
        index = index - length

    return index

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
class TLModel(nn.Module):

    def __init__(self, trained_model):
        super(TLModel, self).__init__()
        # Extract layers from model and remove final layer
        model = list(list(trained_model.children())[0].children())
        cnn = model[:-1]
        linear = model[-1]
        linear = list(linear.children())[:-1]
        new_model = cnn + linear
        new_model = nn.Sequential(*new_model)
        # Freeze all layers before adding embedding layer
        for param in new_model.parameters():
            param.requires_grad = False
        # Add embedding layer and stitch layers together
        self.new_model = nn.Sequential(new_model, nn.Linear(final_feature_dim, embedding_size))
        # Normalize outputs onto hypersphere
        self.norm = Normalize()

    def forward(self, x):
        x = self.new_model(x)
        x = self.norm(x)

        return x

# Triplet loss function for embedder training
class triplet_loss(torch.nn.Module):

    def __init__(self):
        super(triplet_loss, self).__init__()
        # Initialize threshold
        self.alpha = a

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
            start_index = group_indeces.index(label)  # Where to stop loop
            p1 = sorted_by_label[start_index][0]  # First positive
            j = 1  # Index for second positive
            # Find second positive while within same group as first positive
            while group_indeces[wrap_index(n, start_index + j)] == -1:
                i = wrap_index(n, start_index + j)
                stop_after_search = False
                if sorted_by_label[i][1] == label:
                    p2 = sorted_by_label[i][0]
                    ap = torch.dot(p1 - p2, p1 - p2).float()
                    # print("pair index: " + str(i))
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
                        negative = cur_entry[0]
                        # print("negative: " + str(i))
                        an1 = torch.dot(p1 - negative, p1 - negative).float()
                        an2 = torch.dot(p2 - negative, p2 - negative).float()
                        loss = torch.tensor(0).float().cuda()
                        # print(ap, an1, an2)
                        if ap < an1 and ap - an1 + self.alpha > 0:
                            loss += (ap - an1 + self.alpha).float()
                        if ap < an2 and ap - an2 + self.alpha > 0:
                            loss += (ap - an2 + self.alpha).float()
                        losses.append(loss)
                    i = wrap_index(n, i + 1)
                first = False
                j += 1

        #print("finishing loss...")
        bucket_size = 100
        total_loss = torch.tensor(0).float().cuda()
        for i in range(int(math.ceil(float(len(losses))/bucket_size))):
            start = i*bucket_size
            bucket_loss = torch.tensor(0).float().cuda()
            for j in range(bucket_size):
                index = start + j
                if index >= len(losses):
                    break
                bucket_loss += losses[index]
            total_loss += bucket_loss
        return total_loss


def save_obj(fname, obj):
    f = open(fname, 'wb')
    cPickle.dump(obj, f, protocol=2)
    f.close()

old_model = train.Net()
old_model = nn.DataParallel(old_model)
state_dict = torch.load(softmax_checkpoint)
old_model.load_state_dict(state_dict)

net = TLModel(old_model)
net = nn.DataParallel(net)

if use_gpu:
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.cuda()

# if transfer_learning:
#     checkpoint = "experiments/softmax_lr_0001_bs_50_es_3/triplet_checkpoints/999"
#     net.load_state_dict(torch.load(checkpoint))

criterion = triplet_loss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=.0005) #lr = .001

trainset = VehicleDataset.VehicleDataset('train.csv')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=1)

testset = VehicleDataset.VehicleDataset('test.csv', flip=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                        shuffle=True, num_workers=1)



def evaluate(epoch=0):
    print("evaluating")
    net.eval()  #pytorch way of setting the model to eval mode

    # accuracy and magnitudes of pairwise distances
    acc, [same, different] = thresholding(VehicleDataset.VehicleDataset('test.csv', flip=False))
    acc_plot.append(acc)
    save_obj("test_acc/" + str(epoch), acc_plot)

    datasets = []
    plt.figure(figsize=(10,10))
    loaders = [trainset, testset]
    f, axarr = plt.subplots(2, sharex=True)
    for s, set in enumerate(loaders):
        if embedding_size <= 3:
            eval_loader = torch.utils.data.DataLoader(set, batch_size=len(set),
                                                       shuffle=True, num_workers=1)
            with torch.no_grad():
                for i, data in enumerate(eval_loader):
                    images, labels = data
                    if use_gpu:
                        images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    labels = labels
                    datasets.append([outputs, labels])

        binwidth = 5
        bins = int(180/binwidth)
        axarr[s].hist([same, different], bins=bins)
        plt.ylabel(s)

    # Plot triplet loss histograms
    plt.title("Epoch: " + str(epoch))
    plt.savefig("triplet_loss_vis/" + str(epoch) + '.png')
    plt.gcf().clear()
    plt.close()


    print(acc_plot[-1])

    if embedding_size <= 3:
        plot_appearance_vectors(datasets, epoch)
    plt.close()

def thresholding(set):
    net.eval()
    n = 50

    thresholds = [i/20.0 for i in range(1, 31)]
    accs = []
    mags = [[[-1, 0] for j in range(len(testset))] for i in range(len(testset))]
    distances = []
    same_diff = []

    for t, threshold in enumerate(thresholds):

        total = 0.0
        correct = 0.0
        loader = torch.utils.data.DataLoader(set, batch_size=n,
                            shuffle=False, num_workers=1)
        subset = list(loader)[0:n]
        inputs, labels = subset[0]
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)

        for i in range(len(outputs) - 1):
            for j in range(i + 1, len(outputs)):
                if i == j:
                    continue
                if t == 0:
                    target = torch.eq(labels[i], labels[j]).item()
                    diff = outputs[j] - outputs[i]
                    mag = torch.dot(diff, diff)
                    mags[i][j][0] = mag
                    mags[i][j][1] = target
                else:
                    target = mags[i][j][1]
                    mag = mags[i][j][0]
                if mag < threshold:
                    result = 1
                else:
                    result = 0
                if result == target:
                    correct += 1
                total += 1

                distances.append(mag.item())
                same_diff.append(target)
        accs.append([correct/total, threshold])
        same = []
        different = []
        for m in range(len(distances)):
            distance, truth = distances[m], same_diff[m]
            if truth == 0:
                different.append(distance)
            else:
                same.append(distance)

    return [max(accs), [same, different]]



def plot_appearance_vectors(datasets, epoch=0):
    styles = ['r', 'g', 'b', 'y']
    fig = plt.figure()
    for l, dataset in enumerate(datasets):
        ax = fig.add_subplot(211 + l, projection='3d')
        outputs, labels = dataset[0], dataset[1]
        sorted_by_label = zip(outputs, labels)
        sorted_by_label = sorted(sorted_by_label, key=lambda x: x[1])

        outputs, labels = zip(*sorted_by_label)
        #print([label.item() for label in labels])
        prev_label = labels[0]
        indeces = [0]
        for i in range(1, len(labels)):
            label = labels[i]
            if label != prev_label:
                indeces.append(i)
                prev_label = label
        indeces.append(len(labels))
        for i, index in enumerate(indeces):
            if i == len(indeces) - 1:
                break
            xs, ys, zs = [], [], []
            for j in range(index, indeces[i+1]):
                xs.append(outputs[j][0].item())
                ys.append(outputs[j][1].item())
                zs.append(outputs[j][2].item())

                # text = str(x) + ', ' + str(y)
                # plt.text(x, y, text)
            ax.scatter(xs, ys, zs, c=styles[i])
    plt.savefig("spheres/" + str(epoch) + '.png')
    plt.gcf().clear()
    plt.close()

def train():
    net.train(True)

    for epoch in range(1000):  # loop over the dataset multiple times
        evaluate(epoch)
        print("Epoch: " + str(epoch))
        obj_count = 0
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if list(labels.size())[0] != batch_size:
                continue
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # inputs.unsqueeze_(0)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            #print("calculating loss..")
            loss = criterion(outputs, labels)
            loss_plot.append(loss.item())
            running_loss += loss.item()
            if i % 1 == 0:
                running_loss = 0
                print("Loss: " + str(loss.item()))
            if loss != 0:
                loss.backward()
                optimizer.step()
        torch.save(net.state_dict(), "triplet_checkpoints/" + str(epoch))
        save_obj("triplet_losses/" + str(epoch), loss_plot)


def main():
    shutil.rmtree("triplet_checkpoints")
    shutil.rmtree("triplet_loss_vis")
    shutil.rmtree("triplet_losses")
    shutil.rmtree("spheres")
    shutil.rmtree("test_acc")
    os.mkdir("triplet_checkpoints")
    os.mkdir("triplet_loss_vis")
    os.mkdir("triplet_losses")
    os.mkdir("spheres")
    os.mkdir("test_acc")
    train()

if __name__ == "__main__":
    main()
