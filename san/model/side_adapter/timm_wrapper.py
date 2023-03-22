from torch import nn
from timm.models.vision_transformer import _create_vision_transformer
from timm.models import register_model
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding. Modify the original implementation to allow return the 2D patch size."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, (h, w)


@register_model
def vit_w144n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=144, depth=8, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_w192n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=8, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_w240n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=240, depth=8, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_w288n6d8_patch16(pretrained=False, **kwargs):
    assert not pretrained
    model_kwargs = dict(patch_size=16, embed_dim=288, depth=8, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_tiny_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model
