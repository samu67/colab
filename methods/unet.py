import torch
from torch import nn

from common.read_data import *
from common.util import np_to_tensor, accuracy_fn
from common.image_data_set import ImageDataSet
from conv_neural_networks import train


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLu activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(
            2
        )  # pooling layer (can be reused as it will not be trained
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)

        # decode
        for block, upconv, feature in zip(
            self.dec_blocks, self.upconvs, enc_features[::-1]
        ):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip freatures
            x = block(x)  # decrease resolution

        return self.head(x)


def patch_accuracy_fn(y_hat, y):
    # computes accuaracy weighted by patches (metricused on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = (
        y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean(
            (-1, -3)
        )
        < CUTOFF
    )
    patches = (
        y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3))
        < CUTOFF
    )
    return (patches == patches_hat).float().mean()


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = ImageDataSet(
        "data/training", device, use_patches=False, resize_to=(384, 384)
    )
    val_dataset = ImageDataSet(
        "data/validation", device, use_patches=False, resize_to=(384, 384)
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=True
    )
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 35
    train(
        train_dataloader,
        val_dataloader,
        model,
        loss_fn,
        metric_fns,
        optimizer,
        n_epochs,
    )

    # predict on test set
    test_path = "data/test_images/test_images"
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]

    # we also need to resize the test images. This might not be the best idea depending on their spatial resolution
    test_images = np.stack(
        [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0
    )
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)

    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, -1, 1)  # CHW to HWC
    test_pred = np.stack(
        [cv2.resize(img, dsize=size) for img in test_pred], 0
    )  # resize to original

    # now compute labels
    test_pred = test_pred.reshape(
        (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
    )
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)

    create_submission(
        test_pred,
        test_filenames,
        submission_filename="data/submissions/unet_submission.csv",
    )
