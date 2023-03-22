import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import CNNBlockBase, Conv2d
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AddFusion(CNNBlockBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1)
        self.input_proj = nn.Sequential(
            LayerNorm(in_channels),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )
        weight_init.c2_xavier_fill(self.input_proj[-1])

    def forward(self, x: torch.Tensor, y: torch.Tensor, spatial_shape: tuple):
        # x: [N,L,C] y: [N,C,H,W]
        y = (
            F.interpolate(
                self.input_proj(y.contiguous()),
                size=spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(x.shape)
        )
        x = x + y
        return x


def build_fusion_layer(fusion_type: str, in_channels: int, out_channels: int):
    if fusion_type == "add":
        return AddFusion(in_channels, out_channels)
    else:
        raise ValueError("Unknown fusion type: {}".format(fusion_type))
