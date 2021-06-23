import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    absolute_path = os.path.join(ROOT_DIR, path)
    return (
        np.stack(
            [np.array(Image.open(f)) for f in sorted(glob(absolute_path + "/*.png"))]
        ).astype(np.float32)
        / 255.0
    )


def show_first_n(imgs1, imgs2, n=5, title1="Image", title2="Mask"):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(min(n, len(imgs1)), len(imgs2))

    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs1[i])
        axs[1, i].imshow(imgs2[i])
        axs[0, i].set_title(f"{title1} {i}")
        axs[1, i].set_title(f"{title2} {i}")
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.show()


def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array
    # containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (
        w % PATCH_SIZE
    ) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE
    patches = images.reshape(
        (n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1)
    )
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(
            np.maximum(p, np.array([l.item(), 0.0, 0.0]))
        )
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()


# paths to training and validation datasets
train_path = "data/training"
val_path = "data/validation"

train_images = load_all_from_path(os.path.join(train_path, "images"))
train_masks = load_all_from_path(os.path.join(train_path, "groundtruth"))
val_images = load_all_from_path(os.path.join(val_path, "images"))
val_masks = load_all_from_path(os.path.join(val_path, "groundtruth"))

# visualize a few images from the training set
# show_first_n(train_images, train_masks)


# extract all patches and visualize those from the first image
train_patches, train_labels = image_to_patches(train_images, train_masks)
val_patches, val_labels = image_to_patches(val_images, val_masks)

# the first image is broken up in the first 25*25 patches
# show_patched_image(train_patches[:25 * 25], train_labels[:25 * 25])

print(
    "{0:0.2f}".format(sum(train_labels) / len(train_labels) * 100)
    + "% of training patches are labeled as 1."
)
print(
    "{0:0.2f}".format(sum(val_labels) / len(val_labels) * 100)
    + "% of validation patches are labeled as 1."
)


def create_submission(labels, test_filenames, submission_filename):
    test_path = "../data/test_images/test_images"
    with open(os.path.join(ROOT_DIR, submission_filename), "w") as f:
        f.write("id,prediction\n")
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write(
                        "{:03d}_{}_{},{}\n".format(
                            img_number,
                            j * PATCH_SIZE,
                            i * PATCH_SIZE,
                            int(patch_array[i, j]),
                        )
                    )
