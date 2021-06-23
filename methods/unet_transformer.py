import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


from common.read_data import *
from common.util import np_to_tensor, accuracy_fn
from common.image_data_set import ImageDataSet
from conv_neural_networks import train

import math
from math import log, pi, sqrt
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# constants

LayerNorm = partial(nn.InstanceNorm2d, affine = True)
List = nn.ModuleList

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

# positional embeddings

def apply_rotary_emb(q, k, pos_emb):
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]

        seq_x = torch.linspace(-1., 1., steps = h, device = device)
        seq_x = seq_x.unsqueeze(-1)

        seq_y = torch.linspace(-1., 1., steps = w, device = device)
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        scales = self.scales[(*((None,) * (len(seq_y.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi

        x_sinu = repeat(seq_x, 'i d -> i j d', j = w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r = 2), (sin, cos))
        return sin, cos

class TimeSinuPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = einsum('i, j -> i  j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, window_size = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, skip = None, time_emb = None, pos_emb = None):
        h, w, b = self.heads, self.window_size, x.shape[0]

        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb

        q = self.to_q(x)

        kv_input = x

        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim = 0)

        k, v = self.to_kv(kv_input).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) x y c', h = h), (q, k, v))

        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) c -> (b x y) (w1 w2) c', w1 = w, w2 = w), (q, k, v))

        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r = 2), (k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b = b, h = h, y = x.shape[-1] // w, w1 = w, w2 = w)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        hidden_dim = dim * mult
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x, time_emb = None):
        x = self.project_in(x)
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        return self.project_out(x)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        window_size = 16,
        time_emb_dim = None,
        rotary_emb = True
    ):
        super().__init__()
        self.attn_time_emb = None
        self.ff_time_emb = None
        if exists(time_emb_dim):
            self.attn_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            self.ff_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim * ff_mult))

        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None

        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, window_size = window_size)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

    def forward(self, x, skip = None, time = None):
        attn_time_emb = None
        ff_time_emb = None
        if exists(time):
            assert exists(self.attn_time_emb) and exists(self.ff_time_emb), 'time_emb_dim must be given on init if you are conditioning based on time'
            attn_time_emb = self.attn_time_emb(time)
            ff_time_emb = self.ff_time_emb(time)

        pos_emb = None
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(x)

        for attn, ff in self.layers:
            x = attn(x, skip = skip, time_emb = attn_time_emb, pos_emb = pos_emb) + x
            x = ff(x, time_emb = ff_time_emb) + x
        return x

# classes

class Uformer(nn.Module):
    def __init__(
        self,
        dim = 64,
        channels = 3,
        stages = 4,
        num_blocks = 2,
        dim_head = 64,
        window_size = 16,
        heads = 8,
        ff_mult = 4,
        time_emb = False,
        input_channels = None,
        output_channels = None
    ):
        super().__init__()
        input_channels = default(input_channels, channels)
        output_channels = default(output_channels, channels)

        self.to_time_emb = None
        time_emb_dim = None

        if time_emb:
            time_emb_dim = dim
            self.to_time_emb = nn.Sequential(
                TimeSinuPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 3, padding = 1),
            nn.GELU()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding = 1),
        )

        self.downs = List([])
        self.ups = List([])

        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages), (heads, window_size, dim_head, num_blocks))

        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)

            self.downs.append(List([
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
                nn.Conv2d(dim, dim * 2, 4, stride = 2, padding = 1)
            ]))

            self.ups.append(List([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride = 2),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)
            ]))

            dim *= 2

            if is_last:
                self.mid = Block(dim = dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)

    def forward(
        self,
        x,
        time = None
    ):
        if exists(time):
            assert exists(self.to_time_emb), 'time_emb must be set to true to condition on time'
            time = time.to(x)
            time = self.to_time_emb(time)

        x = self.project_in(x)

        skips = []
        for block, downsample in self.downs:
            x = block(x, time = time)
            skips.append(x)
            x = downsample(x)

        x = self.mid(x, time = time)

        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip = skip, time = time)

        x = self.project_out(x)
        return x


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
    model = Uformer(
    dim = 384,           # initial dimensions after input projection, which increases by 2x each stage
    stages = 4,         # number of stages
    num_blocks = 2,     # number of transformer blocks per stage
    window_size = 16,   # set window size (along one side) for which to do the attention within
    dim_head = 64,
    heads = 8,
    ff_mult = 4
    )
    model = model.to(device)
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
        test_pred, test_filenames, submission_filename="data/submissions/unet_submission.csv"
    )


