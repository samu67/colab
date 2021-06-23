import os
from glob import glob
from random import sample

# some constants
VAL_SIZE = 10  # size of the validation set (number of images)

# unzip the dataset, split it and organize it in folders
try:
    for img in sample(glob("training/images/*.png"), VAL_SIZE):
        os.rename(img, img.replace("training", "validation"))
        mask = img.replace("images", "groundtruth")
        os.rename(mask, mask.replace("training", "validation"))
except:
    print("Please upload a .zip file containing")
