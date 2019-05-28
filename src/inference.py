import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import Classifier
import Embedder
import torch.nn.functional as F
import os

use_gpu = True
softmax_checkpoint = "../model_checkpoints/classifier/90v_first_try_78/checkpoints/90v_first_try_78"
embedder_checkpoint = "../model_checkpoints/embedder/current_job/checkpoints/9"

image_size = Classifier.get_image_size()
# Dimensions of embedding layer
embedding_size = Embedder.get_embedding_size()

# Instantiate softmax classifier
softmax = Classifier.Net()
softmax = nn.DataParallel(softmax)
# # Load checkpoint into classifier
# if use_gpu:
#     state_dict = torch.load(softmax_checkpoint)
# else:
#     state_dict = torch.load(softmax_checkpoint, map_location='cpu')
# softmax.load_state_dict(state_dict)
# Instantiate embedding model
embedder = Embedder.Net(softmax)
embedder = nn.DataParallel(embedder)
# Load checkpoint into embedding model
if use_gpu:
    state_dict = torch.load(embedder_checkpoint)
else:
    state_dict = torch.load(embedder_checkpoint, map_location='cpu')
embedder.load_state_dict(state_dict)

# Display number of GPU's being used and push models to GPU(s)
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# Put models in inference mode
softmax.eval()
embedder.eval()

# Returns numpy batch of vectors corresponding to directory of images
def get_vectors_from_crops(dir, model=embedder):
    filenames = [dir + '/' + fname for fname in os.listdir(dir)]
    image_batch = []

    # Open, resize, normalize, and reshape images into network format
    for filename in filenames:
        image = Image.open(filename)
        w, h = image.size
        if w <= image_size:
            resample = Image.NEAREST
        else:
            resample = Image.LANCZOS

        image_resized = np.array(image.resize((image_size, image_size), resample))/255.0
        image_batch.append(image_resized)

    mean = np.mean(image_batch)
    std = np.std(image_batch)
    image_batch = (image_batch - mean) / std
    image_batch = list(image_batch)

    for i, image in enumerate(image_batch):
        image = torch.from_numpy(image.reshape(1, image_size, image_size)).float()
        image_batch[i] = image

    image_batch = torch.stack(image_batch)
    if use_gpu:
        image_batch = image_batch.cuda()
        model = model.cuda()
    outputs = model(image_batch)

    if model == softmax:
        outputs = F.softmax(outputs)

    return outputs.detach().cpu().numpy()

# Returns numpy batch of vectors corresponding to batch of normalized/resized/reshaped numpy images
def get_vectors_from_numpys(ndarray, model=embedder):
    image_batch = torch.from_numpy(ndarray).float()
    if use_gpu:
        image_batch = image_batch.cuda()
        model = model.cuda()
    outputs = model(image_batch)
    
    return outputs.detach().cpu().numpy()

# Returns distances between pairs of hypersphere vectors
def predict_pairwise(outputs):
    mags = []
    for i in range(len(outputs) - 1):
        for j in range(i + 1, len(outputs)):
            output1 = outputs[i]
            output2 = outputs[j]
            diff = output1 - output2
            mags.append(np.dot(diff, diff))
    print(mags)

# Get class predictions from softmax output
def predict_softmax(output_batch):
    predictions = []
    for output in output_batch:
        output = list(output)
        predictions.append(output.index(max(output)))

    return predictions

def main():
    outputs = get_vectors_from_crops("../test_data", model=embedder)
    print(predict_pairwise(outputs))


main()
