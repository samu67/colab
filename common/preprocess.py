import torch
import torchvision.transforms as transforms

from common.read_data import *


def preprocess_test(img):
    to_numpy = False
    if type(img) == np.ndarray:
        to_numpy = True
        img = torch.from_numpy(img)

    img = torch.moveaxis(img, -1, 1)

    t = torch.nn.Sequential(
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.GaussianBlur(kernel_size=11, sigma=10)
    )

    img = torch.moveaxis(t(img), 1, -1)

    if to_numpy:
        img = img.detach().cpu().numpy()

    return img


show_first_n(
    train_images, preprocess_test(train_images), title1="Original", title2="Preprocess"
)
