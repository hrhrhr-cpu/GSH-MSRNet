import argparse
from pathlib import Path

import h5py
import numpy as np


def center_crop(x, shape=(256, 256)):
    h, w = x.shape[-2:]
    target_h, target_w = shape
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot crop {x.shape[-2:]} to {shape}")
    y, z = (h - target_h) // 2, (w - target_w) // 2
    return x[..., y:y + target_h, z:z + target_w]


def rss(kspace):
    h, w = kspace.shape[-2:]
    image = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1))
    image = np.fft.fftshift(image, axes=(-2, -1)) * np.sqrt(h * w)
    return np.sqrt((np.abs(image) ** 2).sum(axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("mask")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("*.h5"))
    if not files:
        raise ValueError(f"No .h5 files found in {args.input_dir}")
    mask = np.load(args.mask).squeeze()

    inputs, targets = [], []
    for path in files:
        with h5py.File(path, "r") as f:
            kspace = f["kspace"][()]
            target = f["reconstruction_rss"][()]

        if kspace.ndim != 4 or target.ndim != 3:
            raise ValueError(f"{path}: unexpected k-space or target shape")
        if mask.shape != kspace.shape[-2:] and mask.shape != (256, 256):
            raise ValueError(f"{path}: incompatible mask shape {mask.shape}")
        if not np.isfinite(kspace).all() or not np.isfinite(target).all():
            raise ValueError(f"{path}: NaN or Inf found")

        kspace = center_crop(kspace)
        current_mask = center_crop(mask)
        observed = rss(kspace * current_mask[None, None])
        target = center_crop(target)

        scale = float(target.max())
        if scale <= 0:
            raise ValueError(f"{path}: empty target volume")

        inputs.append((2.0 * observed / scale - 1.0).astype(np.float32))
        targets.append((2.0 * target / scale - 1.0).astype(np.float32))

    observed = np.concatenate(inputs, axis=0)[:, None]
    target = np.concatenate(targets, axis=0)[:, None]
    if not np.isfinite(observed).all() or not np.isfinite(target).all():
        raise ValueError("Output contains NaN or Inf")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", observed)
    np.save(output_dir / "target.npy", target)
    print(f"Saved {observed.shape} input and {target.shape} target arrays to {output_dir}")


if __name__ == "__main__":
    main()
