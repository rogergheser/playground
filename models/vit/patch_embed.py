from utils import make_2tuple
from typing import Callable, Union, Tuple, Optional
from torch import Tensor

import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """
        Compute the patch embeddings

        Args:
            img_size: Image size
            patch_size: Patch size
            in_chans: Input channels
            embed_dim: Target dimension of the embedding
            norm_layer: Normalisation layer, defaults to None
            flatten_embedding: Whether to keep spatial information or the embeddings
        """
        super().__init__()
        self.img_size = make_2tuple(img_size)
        self.patch_size = make_2tuple(patch_size)
        self.flatten_embedding = flatten_embedding
        self.embed_dim = embed_dim
        patch_grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1]
        )
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape

        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, (
            "Image size should be a multiple of patch size\n"
            f"\tImage size ({H},{W})"
            f"\tPatch size {self.patch_size}"
        )
        x = self.proj(x) # B C H W
        Hp, Wp = x.size(2), x.size(3)

        # Flatten the patch embeddings
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(B, Hp, Wp, self.embed_dim)
        return x

