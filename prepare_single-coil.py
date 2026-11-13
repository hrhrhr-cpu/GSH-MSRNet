import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz")))
    if not files:
        raise ValueError(f"No NIfTI files found in {input_dir}")

    slices = []
    for path in files:
        volume = nib.load(str(path)).get_fdata(dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"{path}: expected a 3-D volume, got {volume.shape}")
        if not np.isfinite(volume).all() or volume.min() < 0:
            raise ValueError(f"{path}: expected finite, non-negative MRI values")

        scale = float(volume.max())
        if scale <= 0:
            raise ValueError(f"{path}: empty volume")

        volume = volume.transpose(2, 1, 0)
        volume = zoom(volume, np.divide((176, 256, 256), volume.shape), order=1)
        slices.append((2.0 * volume / scale - 1.0).astype(np.float32))

    data = np.concatenate(slices, axis=0)[:, None]
    if not np.isfinite(data).all() or data.min() < -1 or data.max() > 1:
        raise ValueError("Output failed the NaN/Inf or [-1,1] range check")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, data)
    print(f"Saved {output}: {data.shape}, float32, {len(files)} volumes")


if __name__ == "__main__":
    main()
