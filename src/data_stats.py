import os
import cv2
import statistics as stat
import numpy as np

data = []

for car in os.listdir("crops"):
    for image_name in os.listdir("crops/" + car):
        image = cv2.imread("crops/" + car + '/' + image_name, 0)
        try:
            side = image.shape[::-1][0]
        except:
            print("Invalid image...")
            continue
        data.append(side)

mean = stat.mean(data)
std = stat.pstdev(data)

print(mean, std)

