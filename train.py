import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from model import Discriminator, Generator

SEED = 42
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
BETAS = (0.9, 0.999)
LR_DECAY = 0.5
LR_DECAY_EVERY = 10
PATIENCE = 10
W_PIXEL = 15.0
W_FREQ = 0.1
W_PERC = 0.0025
W_ADV = 1.0
IMAGE_SIZE = 256


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MRIDataset(Dataset):
    def __init__(self, path):
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            data = np.load(path)
        data = np.asarray(data)
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]
        elif data.ndim == 4 and data.shape[-1] == 1:
            data = data.transpose(0, 3, 1, 2)
        if data.ndim != 4 or data.shape[1] != 1:
            raise ValueError("data must have shape [N, H, W] or [N, H, W, 1]")
        if data.shape[2] != IMAGE_SIZE or data.shape[3] != IMAGE_SIZE:
            raise ValueError("images must be 256 x 256")
        if float(data.min()) < -1.0 or float(data.max()) > 1.0:
            raise ValueError("images must be normalized to [-1, 1]")
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.features = models.vgg16(weights=weights).features[:-7].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.resize = transforms.Resize((244, 244))
        self.register_buffer(
            "mean", torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x + 1.0) * 127.5
        x = x - self.mean
        x = self.resize(x)
        return self.features(x)


def gaussian_mask(size, rate, seed):
    rng = np.random.default_rng(seed)
    coords = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2) / (size // 2)
    density = np.exp(-(radius**2) / (2 * 0.35**2))
    core = radius <= 0.05
    mask = core.astype(np.float32)
    target = int(round(rate * size * size))
    remaining = target - int(mask.sum())
    if remaining > 0:
        weights = np.where(core, 0.0, density)
        weights = weights.ravel() / weights.sum()
        available = np.flatnonzero(~core.ravel())
        chosen = rng.choice(available, size=remaining, replace=False, p=weights[available])
        mask.ravel()[chosen] = 1.0
    return torch.from_numpy(mask).view(1, 1, size, size)


def undersample(x, mask):
    x01 = (x + 1.0) / 2.0
    kspace = torch.fft.fftshift(
        torch.fft.fft2(x01, dim=(-2, -1)), dim=(-2, -1)
    )
    kspace = kspace * mask
    image = torch.abs(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)
        )
    )
    return image * 2.0 - 1.0


def nmse(x, y):
    numerator = F.mse_loss(x, y)
    denominator = F.mse_loss(y, torch.zeros_like(y))
    return numerator / denominator


def frequency_loss(x, y):
    x01 = (x + 1.0) / 2.0
    y01 = (y + 1.0) / 2.0
    xf = torch.abs(torch.fft.fftn(x01, dim=(-2, -1)))
    yf = torch.abs(torch.fft.fftn(y01, dim=(-2, -1)))
    return F.mse_loss(xf, yf)


def validate(generator, loader, mask, device):
    generator.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for ground_truth in loader:
            ground_truth = ground_truth.to(device)
            observed = undersample(ground_truth, mask)
            reconstructed = generator(observed)
            total += nmse(reconstructed, ground_truth).item() * ground_truth.size(0)
            count += ground_truth.size(0)
    generator.train()
    return total / count


def save_checkpoint(path, generator, discriminator):
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = MRIDataset(args.train_data)
    val_set = MRIDataset(args.val_data)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    if args.mask is None:
        mask = gaussian_mask(IMAGE_SIZE, 0.10, SEED).to(device)
    else:
        mask_array = np.load(args.mask).astype(np.float32)
        mask = torch.from_numpy(mask_array).view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg = VGGFeatureExtractor().to(device)

    bce = nn.BCEWithLogitsLoss()
    g_optim = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)
    g_scheduler = torch.optim.lr_scheduler.StepLR(
        g_optim, step_size=LR_DECAY_EVERY * len(train_loader), gamma=LR_DECAY
    )
    d_scheduler = torch.optim.lr_scheduler.StepLR(
        d_optim, step_size=LR_DECAY_EVERY * len(train_loader), gamma=LR_DECAY
    )

    best_nmse = float("inf")
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    last_path = os.path.join(args.out_dir, "last_model.pt")

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        running_g = 0.0
        running_d = 0.0
        batches = 0
        for ground_truth in train_loader:
            ground_truth = ground_truth.to(device)
            observed = undersample(ground_truth, mask)
            reconstructed = generator(observed)
            _, logits_real = discriminator(ground_truth)
            _, logits_fake = discriminator(reconstructed)
            real_target = torch.ones_like(logits_real)
            fake_target = torch.zeros_like(logits_fake)
            d_loss = bce(logits_real, real_target) + bce(logits_fake, fake_target)
            g_adv = bce(logits_fake, torch.ones_like(logits_fake))
            g_pixel = nmse(reconstructed, ground_truth)
            g_freq = frequency_loss(reconstructed, ground_truth)
            g_perc = F.mse_loss(vgg(reconstructed), vgg(ground_truth))
            g_loss = (
                W_ADV * g_adv
                + W_PIXEL * g_pixel
                + W_FREQ * g_freq
                + W_PERC * g_perc
            )
            d_optim.zero_grad()
            d_loss.backward(retain_graph=True)
            g_optim.zero_grad()
            g_loss.backward()
            d_optim.step()
            g_optim.step()
            d_scheduler.step()
            g_scheduler.step()
            running_d += d_loss.item()
            running_g += g_loss.item()
            batches += 1

        val_nmse = validate(generator, val_loader, mask, device)
        print(
            "epoch {:03d} g_loss {:.6f} d_loss {:.6f} val_nmse {:.8f}".format(
                epoch + 1,
                running_g / batches,
                running_d / batches,
                val_nmse,
            ),
            flush=True,
        )
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            bad_epochs = 0
            save_checkpoint(best_path, generator, discriminator)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping at epoch {:03d}".format(epoch + 1), flush=True)
                break

    save_checkpoint(last_path, generator, discriminator)


if __name__ == "__main__":
    main()
