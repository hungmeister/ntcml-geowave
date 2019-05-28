import matplotlib.pyplot as plt
import pickle as cPickle
import random
import argparse

def load_obj(fname):
    f = open(fname, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()

    return loaded_obj

def plot_classifier_logs(filename):
    logs = load_obj(filename)

    loss = logs["Loss"]
    test_acc = logs["Test Accuracy"]
    train_acc = logs["Train Accuracy"]

    plt.figure(1)
    plt.subplot(211)
    plt.ylabel('Loss')
    plt.plot([i for i in range(len(loss[0])*len(loss))], [item for sublist in loss for item in sublist])

    plt.subplot(212)
    plt.ylabel('Accuracy')
    x_log = [i for i in range(len(test_acc))]
    plt.plot(x_log, train_acc, 'g', x_log, test_acc, 'b')
    plt.show()

def plot_embedding_logs(filename):
    epoch = int(input("Enter epoch for L2 distribution:"))

    logs = load_obj(filename)

    loss = logs['Loss']
    test_eer = logs['Test EER']
    train_eer = logs['Train EER']
    test_dist = logs['Test Distribution']
    train_dist = logs['Train Distribution']

    plt.figure(1)
    plt.subplot(411)
    plt.ylabel('Loss')
    plt.plot([i for i in range(len(loss[0]) * len(loss))], [item for sublist in loss for item in sublist])

    plt.subplot(412)
    plt.ylabel('EER')
    plt.plot([i for i in range(len(test_eer))], test_eer, 'g', train_eer, 'b')

    plt.subplot(413)
    plt.hist(train_dist[epoch][0], color='g', bins=30)

    plt.subplot(414)
    plt.hist(train_dist[epoch][1], color='r', bins=30)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="specify whether softmax or embedding log file")
    parser.add_argument("filename", help="specify location of log file")
    args = parser.parse_args()
    filename = str(args.filename)
    type = int(args.type)

    if type == 0:
        plot_classifier_logs(filename)
    elif type == 1:
        plot_embedding_logs(filename)
    else:
        print(type, "is not a valid log type!")

