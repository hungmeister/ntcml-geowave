import pickle
import torch
import os
import Classifier
import Embedder

# Pickle some object
def save_obj(fname, obj):
    f = open(fname, 'wb')
    pickle.dump(obj, f, protocol=2)
    f.close()

def load_obj(fname):

    return pickle.load(open(fname, 'rb'))

def load_state(model, filename, optimizer=None):
    state = torch.load(filename)
    model.load_state_dict(state["Checkpoint"])
    if optimizer != None:
        optimizer.load_state_dict(state["Optimizer"])

        return [model, optimizer]
    else:
        return model

def save_state(dest, model, optimizer, filename):
    state = {"Checkpoint": model.state_dict(), "Optimizer": optimizer.state_dict()}
    torch.save(state, os.path.join(dest, filename))