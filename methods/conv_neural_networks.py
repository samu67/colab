import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import dataloader
from tqdm.notebook import tqdm

from common.util import np_to_tensor, accuracy_fn
from common.read_data import *
from common.image_data_set import ImageDataSet


def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)] * 3, -1))
            axs[0, i].set_title(f"Sample {i}")
            axs[1, i].set_title(f"Predicted {i}")
            axs[2, i].set_title(f"True {i}")
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
    else:
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(
                f"True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}"
            )
    plt.show()


def train(
    train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs
):
    # traininng loop
    logdir = os.path.join(ROOT_DIR, "data/tensorboard")
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}

    for epoch in range(n_epochs):

        # initialize metric list
        metrics = {"loss": [], "val_loss": []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics["val_" + k] = []

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")

        # training
        model.train()
        for x, y in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            # log partial metrics
            metrics["loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix(
                {k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0}
            )

        # validation
        model.eval()
        with torch.no_grad():
            for (x, y) in eval_dataloader:
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                metrics["val_loss"].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tennsorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        print(
            " ".join(
                [
                    "\t- " + str(k) + " = " + str(v) + "\n "
                    for (k, v) in history[epoch].items()
                ]
            )
        )
        show_val_samples(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            y_hat.detach().cpu().numpy(),
        )

    print("Finished Training")
    # plot loss curves
    plt.plot([v["loss"] for k, v in history.items()], label="Training Loss")
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


class PatchCNN(nn.Module):

    # simple CNN for classification of patches
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = ImageDataSet("data/training", device)
    val_dataset = ImageDataSet("data/validation", device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True
    )
    model = PatchCNN().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {"acc": accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 20
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
    test_path = os.path.join(ROOT_DIR, "data/test_images/test_images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_patches = np.moveaxis(image_to_patches(test_images), -1, 1)  # HWC to CHW
    test_patches = np.reshape(
        test_patches, (38, -1, 3, PATCH_SIZE, PATCH_SIZE)
    )  # split in batches for memory constraints
    test_pred = [
        model(np_to_tensor(batch, device)).detach().cpu().numpy()
        for batch in test_patches
    ]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.round(
        test_pred.reshape(
            test_images.shape[0],
            test_images.shape[1] // PATCH_SIZE,
            test_images.shape[1] // PATCH_SIZE,
        )
    )

    create_submission(
        test_pred,
        test_filenames,
        submission_filename="data/submissions/cnn_submission.csv",
    )
