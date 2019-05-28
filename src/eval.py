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
import shutil
import os
import random
import Embedder
import Classifier
from config import USE_GPU

eval_thresholds = [_/float(10) for _ in range(1, 40)]
eval_size = 50

# Load some pickled object
def load_obj(fname):
    f = open(fname, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()

    return loaded_obj

# Pickle some object
def save_obj(fname, obj):
    f = open(fname, 'wb')
    cPickle.dump(obj, f, protocol=2)
    f.close()

# get the roc curve and the L2 distributions for the given model and dataset
def roc_dist(model, set, n):
    print("in roc_dist method")
    model.eval()

    mags = [[[-1, 0] for j in range(n)] for i in range(n)]
    distances = []
    same_diff = []

    loader = torch.utils.data.DataLoader(set, batch_size=n,
                                         shuffle=True, num_workers=1)

    num_neg = 0
    num_pos = 0
    tprs = []
    fprs = []
    ts = []
    for t, threshold in enumerate(eval_thresholds):
        inputs, labels = list(loader)[0]
        if USE_GPU:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        tp = 0
        fp = 0
        for i in range(len(outputs) - 1):
            for j in range(i + 1, len(outputs)):
                if i == j:
                    continue
                if t == 0:
                    target = torch.eq(labels[i], labels[j]).item()
                    if target == 1:
                        num_pos += 1
                    else:
                        num_neg += 1
                    diff = outputs[j] - outputs[i]
                    mag = torch.dot(diff, diff)
                    mags[i][j][0] = mag
                    mags[i][j][1] = target
                else:
                    target = mags[i][j][1]
                    mag = mags[i][j][0]
                if mag.item() < threshold:
                    result = 1
                else:
                    result = 0
                # True positive
                if target == 1:
                    if target == result:
                        tp += 1
                # False positive
                else:
                    if target != result:
                        fp += 1

                distances.append(mag.item())
                same_diff.append(target)
        tprs.append(float(tp)/num_pos)
        fprs.append(float(fp)/num_neg)
        ts.append(threshold)
        same = []
        different = []
        for m in range(len(distances)):
            distance, truth = distances[m], same_diff[m]
            if truth == 0:
                different.append(distance)
            else:
                same.append(distance)

    return [[ts, np.array(tprs), np.array(fprs)], [same, different]]

# Get the EER for a given roc curve
def eer(roc):
    ts, tprs, fprs = roc
    fnrs = np.array([1 for _ in tprs]) - tprs
    argmin = np.argmin(np.absolute(fnrs - fprs))
    t = ts[argmin]
    fpr = fprs[argmin]
    fnr = fnrs[argmin]

    print("EER (fpr, fnr):", fpr, fnr)

    # plt.figure()
    # plt.plot(fprs, tprs)
    # plt.show()

    return [fpr, t]

# Call roc_dist on train and test sets, and log all relevant statistics
def log_embedding(model, trainset, testset, losses, filename):
    print("in eval log_embedding method")
    model.eval()

    train_roc, train_dist = roc_dist(model, trainset, eval_size)
    print("thresholded train")
    test_roc, test_dist = roc_dist(model, testset, eval_size)
    print("thresholded test")

    # IMPLEMENT EER
    train_eer, t_train = eer(train_roc)
    test_eer, t_test = eer(test_roc)


    if os.path.exists(filename):
        log_dict = load_obj(filename)
        log_dict["Loss"] = np.concatenate((log_dict["Loss"], np.asarray([losses])))
        log_dict["Test EER"] = np.concatenate((log_dict["Test EER"], np.asarray([test_eer])))
        log_dict["Train EER"] = np.concatenate((log_dict["Train EER"], np.asarray([train_eer])))
        log_dict["Test Distribution"] = np.concatenate((log_dict["Test Distribution"], np.asarray([test_dist])))
        log_dict["Train Distribution"] = np.concatenate((log_dict["Train Distribution"], np.asarray([train_dist])))
    else:
        log_dict = {"Loss": np.asarray([losses]),
                    "Test EER": np.asarray([test_eer]),
                    "Train EER": np.asarray([train_eer]),
                    "Test Distribution": np.asarray([test_dist]),
                    "Train Distribution": np.asarray([train_dist])
                    }

    save_obj(filename, log_dict)

# Get the classification accuracy for a given model and dataset
def classification_acc(model, set, n=0):
    if n == 0:
        n = len(set)
    model.eval()
    loader = torch.utils.data.DataLoader(set, batch_size=n,
                                                   shuffle=True, num_workers=1)
    net_correct = 0
    net_total = 0
    for inputs, labels in loader:
        if USE_GPU:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        predictions = []
        for output in outputs:
            output = list(output)
            predictions.append(output.index(max(output)))
        total = len(predictions)
        correct = [1 if predictions[i] - list(labels.detach().cpu().numpy())[i] == 0 else 0 for i in range(total)]
        correct = sum(correct)
        net_correct += correct
        net_total += total

    return float(net_correct)/net_total

# Call classification_acc on train and test sets and log relevant statistics
def log_classifier(model, trainset, testset, losses, filename):
    model.eval()
    # n = len(testset)
    n = 100

    train_acc = classification_acc(model, trainset, n=n)
    test_acc = classification_acc(model, testset, n=n)

    if os.path.exists(filename):
        log_dict = load_obj(filename)
        log_dict["Loss"] = np.concatenate((log_dict["Loss"], np.asarray([losses])))
        log_dict["Test Accuracy"] = np.concatenate((log_dict["Test Accuracy"], np.asarray([test_acc])))
        log_dict["Train Accuracy"] = np.concatenate((log_dict["Train Accuracy"], np.asarray([train_acc])))
    else:
        log_dict = {"Loss": np.asarray([losses]),
                    "Test Accuracy": np.asarray([test_acc]),
                    "Train Accuracy": np.asarray([train_acc])
                    }

    print(test_acc, train_acc)
    save_obj(filename, log_dict)

