import torch
import torch.nn as nn
from torch import Tensor


class HalfConv(nn.Module):
    def __init__(self, dim: int, n_div: int = 2, kernel_size: int = 3):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_skip = dim - self.dim_conv
        self.partial_conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_conv, x_skip = torch.split(x, [self.dim_conv, self.dim_skip], dim=1)
        x_conv = self.partial_conv(x_conv)
        return torch.cat((x_conv, x_skip), dim=1)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class SEGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class DWConvPW(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return self.act(x)


class GSHalfConv(nn.Module):
    def __init__(
        self,
        dim: int,
        groups: int = 3,
        n_div: int = 2,
        se_reduction: int = 4,
        shuffle_groups: int = 3,
        dw_kernel: int = 3,
    ):
        super().__init__()
        base = dim // groups
        remainder = dim - base * (groups - 1)
        group_dims = [base] * (groups - 1) + [remainder]
        self.group_blocks = nn.ModuleList(
            [HalfConv(d, n_div=n_div, kernel_size=3) for d in group_dims]
        )
        self.groups = groups
        self.shuffle_groups = shuffle_groups if dim % shuffle_groups == 0 else 1
        self.mix = DWConvPW(dim, kernel_size=dw_kernel)
        self.se = SEGate(dim, reduction=se_reduction)
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        base = x.shape[1] // self.groups
        remainder = x.shape[1] - base * (self.groups - 1)
        group_dims = [base] * (self.groups - 1) + [remainder]
        xs = torch.split(x, group_dims, dim=1)
        outs = [blk(t) for t, blk in zip(xs, self.group_blocks)]
        out = torch.cat(outs, dim=1)
        out = channel_shuffle(out, self.shuffle_groups)
        out = self.mix(out)
        out = self.se(out)
        out = self.norm(out)
        return identity + self.res_scale * out


class MSRRevConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 5,
        scale: int = 1,
        padding: int = 2,
        padding_mode: str = "circular",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.weight_main = nn.Parameter(
            torch.randn(1, in_channels, kernel_size, kernel_size)
        )
        self.weight_large = nn.Parameter(torch.randn(1, in_channels, 7, 7))
        self.weight_small = nn.Parameter(torch.randn(1, in_channels, 3, 3))
        self.kernel_alpha = nn.Parameter(torch.ones(3))
        self.lam = nn.Parameter(torch.full((1, in_channels, 1, 1), 1e-3))
        with torch.no_grad():
            self.weight_main.copy_(self._softmax_kernel(self.weight_main))
            self.weight_large.copy_(self._softmax_kernel(self.weight_large))
            self.weight_small.copy_(self._softmax_kernel(self.weight_small))

    def forward(self, x: Tensor) -> Tensor:
        if self.padding > 0:
            x = nn.functional.pad(
                x,
                pad=[self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
                value=0,
            )
        _, _, h, w = x.shape
        sty = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x_up = nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        else:
            x_up = x
        fb_main = self.p2o(self.weight_main, (h * self.scale, w * self.scale))
        fb_large = self.p2o(self.weight_large, (h * self.scale, w * self.scale))
        fb_small = self.p2o(self.weight_small, (h * self.scale, w * self.scale))
        alphas = torch.softmax(self.kernel_alpha, dim=0)
        fb = alphas[0] * fb_main + alphas[1] * fb_large + alphas[2] * fb_small
        fbc = torch.conj(fb)
        f2b = torch.pow(torch.abs(fb), 2)
        fbfy = fbc * torch.fft.fftn(sty, dim=(-2, -1))
        fr = fbfy + torch.fft.fftn(self.lam * x_up, dim=(-2, -1))
        x1 = fb.mul(fr)
        fbr = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        inv_w = torch.mean(self.splits(f2b, self.scale), dim=-1, keepdim=False)
        inv_wbr = fbr.div(inv_w + self.lam)
        fcb_inv_wbr = fbc * inv_wbr.repeat(1, 1, self.scale, self.scale)
        fx = (fr - fcb_inv_wbr) / self.lam
        out = torch.real(torch.fft.ifftn(fx, dim=(-2, -1)))
        if self.padding > 0:
            out = out[
                ...,
                self.padding * self.scale : -self.padding * self.scale,
                self.padding * self.scale : -self.padding * self.scale,
            ]
        return out

    def splits(self, a: Tensor, scale: int) -> Tensor:
        *leading_dims, width, height = a.size()
        width_s, height_s = width // scale, height // scale
        b = a.view(*leading_dims, scale, width_s, scale, height_s)
        permute_order = list(range(len(leading_dims))) + [
            len(leading_dims) + 1,
            len(leading_dims) + 3,
            len(leading_dims),
            len(leading_dims) + 2,
        ]
        b = b.permute(*permute_order).contiguous()
        return b.view(*leading_dims, width_s, height_s, scale * scale)

    def p2o(self, psf: Tensor, shape) -> Tensor:
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., : psf.shape[-2], : psf.shape[-1]].copy_(psf)
        otf = torch.roll(
            otf,
            (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)),
            dims=(-2, -1),
        )
        return torch.fft.fftn(otf, dim=(-2, -1))

    def upsample(self, x: Tensor, scale: int = 3) -> Tensor:
        z = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)
        ).type_as(x)
        z[..., ::scale, ::scale].copy_(x)
        return z

    @staticmethod
    def _softmax_kernel(k: Tensor) -> Tensor:
        b, c, h, w = k.shape
        k_ = k.view(b, c, -1)
        k_ = torch.softmax(k_, dim=-1)
        return k_.view(b, c, h, w)


class RIRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = self._conv_bn_lrelu(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = self._half_bn_lrelu(mid_channels)
        self.conv3 = self._half_bn_lrelu(mid_channels)
        self.conv4 = self._conv_bn_lrelu(
            mid_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    @staticmethod
    def _conv_bn_lrelu(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _half_bn_lrelu(channels: int) -> nn.Sequential:
        return nn.Sequential(
            GSHalfConv(dim=channels),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        rir_in = conv1_out + conv3_out
        conv4_out = self.conv4(rir_in)
        return x + conv4_out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.RIR = RIRBlock(out_channels, out_channels)
        self.conv_b = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_a(x)
        x = self.RIR(x)
        return self.conv_b(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv_a = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.rir_block = RIRBlock(out_channels, out_channels)
        self.deconv_b = MSRRevConv(
            in_channels=out_channels, kernel_size=5, scale=2, padding=2
        )

    def forward(self, x: Tensor) -> Tensor:
        deconv_a_out = self.deconv_a(x)
        rir_out = self.rir_block(deconv_a_out)
        return self.deconv_b(rir_out)


class Generator(nn.Module):
    def __init__(self, nb_filters: int = 64):
        super().__init__()
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder1_1 = EncoderBlock(nb_filters, nb_filters)
        self.encoder1_2 = EncoderBlock(nb_filters, 2 * nb_filters)
        self.encoder1_3 = EncoderBlock(2 * nb_filters, 4 * nb_filters)
        self.encoder1_4 = EncoderBlock(4 * nb_filters, 8 * nb_filters)
        self.decoder1_4 = DecoderBlock(8 * nb_filters, 4 * nb_filters)
        self.decoder1_3 = DecoderBlock(4 * nb_filters, 2 * nb_filters)
        self.decoder1_2 = DecoderBlock(2 * nb_filters, nb_filters)
        self.decoder1_1 = DecoderBlock(nb_filters, nb_filters)
        self.output1_1 = nn.Sequential(
            nn.Conv2d(nb_filters, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1_0(x)
        x1 = self.encoder1_1(x)
        x2 = self.encoder1_2(x1)
        x3 = self.encoder1_3(x2)
        x4 = self.encoder1_4(x3)
        d1 = self.decoder1_4(x4)
        d2 = self.decoder1_3(d1 + x3)
        d3 = self.decoder1_2(d2 + x2)
        d4 = self.decoder1_1(d3 + x1)
        return self.output1_1(d4 + x)


class Discriminator(nn.Module):
    def __init__(self, df_filters: int = 64):
        super().__init__()
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=df_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )
        self.conv1_1 = EncoderBlock(df_filters, df_filters)
        self.conv1_2 = EncoderBlock(df_filters, df_filters * 2)
        self.conv1_3 = EncoderBlock(df_filters * 2, df_filters * 4)
        self.conv1_4 = EncoderBlock(df_filters * 4, df_filters * 8)
        self.conv1_5 = nn.Conv2d(
            in_channels=df_filters * 8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 1, bias=False)

    def forward(self, x: Tensor):
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.conv1_5(x)
        x = self.flatten(x)
        logits = self.fc(x)
        probs = torch.sigmoid(logits)
        return probs, logits
