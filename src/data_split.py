import os
import random
from shutil import copyfile
from PIL import Image
import csv
import pandas as pd
import xml.etree.cElementTree as ET

image_format = ".tif"
test_xml = []
train_xml = []
csv_names = ["test.csv", "train.csv"]
size = 20
id_mapping = []

for car in os.listdir("crops"):
    crops = os.listdir("crops/" + car)
    random.shuffle(crops)
    n = len(crops)
    test_size = int(round(n/10))
    test = crops[:test_size]
    train = crops[test_size:]
    sets = [["train/", train, train_xml], ["test/", test, test_xml]]

    if car not in id_mapping:
        id_mapping.append(car)
    f = open("class_mapping.txt", 'w')
    for i, id in enumerate(id_mapping):
        f.write(str(id) + ":" + str(i) + '\n')


    for set in sets:
        for i, crop in enumerate(set[1]):
            image = Image.open("crops/" + car + '/' + crop)
            w, h = image.size

            if w <= size:
                resample = Image.NEAREST
            else:
                resample = Image.LANCZOS

            image_resized = image.resize((size, size), resample)
            if car not in os.listdir(set[0]):
                os.mkdir(set[0] + car)
            image_resized.save(set[0] + car + '/' + crop)
            set[2].append([set[0] + car + '/' + crop, id_mapping.index(car)])

for i, xml_list in enumerate([test_xml, train_xml]):
    column_name = ['image name', 'label']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(csv_names[i], index=None)
