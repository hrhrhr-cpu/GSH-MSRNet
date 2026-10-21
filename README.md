# GSH-MSRNet: minimal reproduction code

This package contains the generator, the discriminator, and the training procedure of GSH-MSRNet. Comparison models and sampling-mask files are not included.

## Files

| File | Content |
|---|---|
| `model.py` | GSH-MSRNet generator and discriminator |
| `train.py` | Training and validation procedure, seed, optimizer, scheduler, losses, early stopping |
| `requirements.txt` | Python dependencies |

## Environment

Tested with Python 3.8.18, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, NumPy 1.24.4, on an NVIDIA GeForce RTX 3090.

## Data format

`train.py` accepts `.npy`, `.pkl`, or `.pickle` files containing a single array:

- shape `[N, 256, 256, 1]`, or `[N, 256, 256]`, or `[N, 1, 256, 256]`
- single-channel magnitude images, cropped or padded to 256 x 256
- values normalized to `[-1, 1]`
- training and validation sets stored in separate files

The experiments used a 70/10/20 training/validation/test split. The test set is not needed for training.

## Run

```bash
python train.py --train_data /path/to/train.npy --val_data /path/to/val.npy --out_dir outputs
```

Optional arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--mask` | none | binary mask of shape `[256, 256]` or `[1, 1, 256, 256]`; `.npy` |
| `--out_dir` | `outputs` | directory for checkpoints |
| `--num_workers` | `0` | DataLoader worker count |

When `--mask` is omitted, the script generates a deterministic 10% Gaussian variable-density mask with seed 42. The published experiments used the dataset's Gaussian 2-D variable-density mask; supply that mask with `--mask` if the exact sampling pattern is required.

Checkpoints are written to `outputs/best_model.pt` and `outputs/last_model.pt`. Each file stores the generator and discriminator state dictionaries under the keys `generator` and `discriminator`.

Loading the trained generator:

```python
import torch
from model import Generator

checkpoint = torch.load("outputs/best_model.pt", map_location="cpu")
generator = Generator()
generator.load_state_dict(checkpoint["generator"])
generator.eval()
```

## Architecture settings

Generator:

| Item | Setting |
|---|---|
| Input / output | 1 x 256 x 256 / 1 x 256 x 256, range `[-1, 1]` |
| Base width | 64 |
| Encoder | four stages, channels 64 / 128 / 256 / 512, stride-2 convolution downsampling |
| Bottleneck | 512 x 16 x 16 |
| Decoder | four stages, channels 256 / 128 / 64 / 64 |
| Decoder upsampling | MSRRevConv, scale 2, padding 2 |
| Skips | element-wise addition |
| Output layer | 3 x 3 convolution followed by Tanh |
| Parameters | 10,818,207 |

GSHalfConv defaults: `groups=3`, `n_div=2`, `se_reduction=4`, `shuffle_groups=3`, `dw_kernel=3`, followed by batch normalization and a learnable residual scale. HalfConv applies a 3 x 3 convolution to half of the channels and passes the other half through unchanged. Channel shuffle reduces to the identity map when the channel count is not divisible by the shuffle group count.

MSRRevConv: three depth-wise kernels of sizes 5 x 5, 7 x 7, and 3 x 3, fused in the Fourier domain; circular padding of 2 pixels; upsampling scale 2; channel-wise trainable regularization parameter `lambda`.

Discriminator: one input channel; stride-2 stem with 64 channels; four encoder stages with 64 / 128 / 256 / 512 channels; a 3 x 3 convolution to one channel; flatten; linear layer to one logit. Input is 256 x 256, so the flattened feature has 64 elements. The discriminator reuses the same GSHalfConv/RIRBlock machinery as the generator encoder.

## Initialization

The seed is fixed to 42 before model construction, data loading, and training:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Initialization follows the released model:

- all layers use the PyTorch default initialization except the items below;
- `GSHalfConv.res_scale` is initialized to 0.5;
- the three MSRRevConv kernels are drawn from `torch.randn` and then softmax-normalized over their spatial support, so each kernel starts as a spatial probability distribution;
- MSRRevConv `kernel_alpha` is initialized to ones, giving uniform fusion weights of 1/3 before the softmax;
- MSRRevConv `lambda` is a raw trainable parameter initialized to `1e-3` per channel, with no positivity transform (no Softplus);
- the VGG16 feature extractor uses ImageNet-pretrained weights and is frozen.

## Training hyperparameters

| Item | Setting |
|---|---|
| Optimizer | Adam for generator and discriminator |
| Learning rate | `1e-4` |
| Betas | `(0.9, 0.999)` |
| Scheduler | StepLR, gamma 0.5 every 10 epochs |
| Batch size | 8 |
| Maximum epochs | 50 |
| Pixel loss weight | 15 |
| Frequency loss weight | 0.1 |
| Perceptual loss weight | 0.0025 |
| Adversarial loss weight | 1 |
| Discriminator loss | BCEWithLogits against real=1 and fake=0 |
| Perceptual features | VGG16 `features[:-7]`, input resized to 244 x 244 |

Generator objective:

```text
L_G = 1.0 * L_adv + 15 * L_pixel + 0.1 * L_freq + 0.0025 * L_perceptual
```

`L_pixel` is NMSE, `L_freq` is the MSE between the magnitudes of the 2-D Fourier transforms of the reconstructed and ground-truth images, and `L_perceptual` is the MSE between VGG16 conv4 features. The generator and discriminator are updated simultaneously: the discriminator backward pass is retained, then the generator backward pass is computed, and both optimizers step once.

The scheduler is stepped once per optimization step with `step_size = 10 * len(train_loader)`, which is equivalent to decaying every 10 epochs.

## Stopping criterion

After each epoch, the generator is evaluated on the validation set by NMSE. The best validation NMSE is tracked, and the best generator/discriminator pair is saved. Training stops when validation NMSE has not improved for 10 consecutive epochs, or after 50 epochs.

## Citation

If you use this code, please cite the GSH-MSRNet paper.
