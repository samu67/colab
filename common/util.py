import torch


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    else:
        return (
            torch.from_numpy(x)
            .contiguous()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()
