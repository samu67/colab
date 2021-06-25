import torch

from common.util import np_to_tensor
from common.read_data import *


class ImageDataSet(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index

    def __init__(self, path, device, use_patches=True, resize_to=(192, 192)):
        self.path = path

        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, "images"))
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        else:
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x])
            self.y = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.y])
        self.x = np.moveaxis(self.x, -1, 1)
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing

        s = torch.std(x, [1, 2])

        for i in range(0, x.shape[1], 16):
            for j in range(0, x.shape[1], 16):
                m = torch.mean(x[:, i : i + 16, j : j + 16], [1, 2])

                x[0, i : i + 16, j : j + 16] /= m[0]
                x[1, i : i + 16, j : j + 16] /= m[1]
                x[2, i : i + 16, j : j + 16] /= m[2]

        x[0] /= s[0]
        x[1] /= s[1]
        x[2] /= s[2]

        return x, y

    def __getitem__(self, item):
        return self._preprocess(
            np_to_tensor(self.x[item], self.device),
            np_to_tensor(self.y[[item]], self.device),
        )

    def __len__(self):
        return self.n_samples
