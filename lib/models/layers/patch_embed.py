import torch.nn as nn
import torch
from timm.models.layers import to_2tuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, stride_size = 16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride= stride_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.level_s = 4
        self.special_weight = nn.Parameter(torch.zeros((self.level_s + 1), 1, embed_dim, 1, 1))
        torch.nn.init.kaiming_uniform_(self.special_weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.dim = embed_dim

    def forward(self, zx_feat, if_smooth):
        if if_smooth:         
            x_all = self.proj(zx_feat) 
            B, C, H, W = x_all.shape     
           
            x = x_all.view(self.level_s + 1, B//(self.level_s + 1), self.dim, H, W)

            weight = torch.nn.functional.softmax(self.special_weight, dim=0)
            weight =  weight.expand(self.level_s + 1, B//(self.level_s + 1), self.dim, H, W)
                   
            x = x * weight
            x = torch.sum(x, dim=0)

            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

            x = self.norm(x)

            x = x.transpose(1, 2).contiguous().view(B//(self.level_s + 1), C, H, W)

            return x
            
        else:
            x = self.proj(zx_feat)

            B, C, H, W = x.shape
    
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

            x = self.norm(x)

            x = x.transpose(1, 2).contiguous().view(B, C, H, W)
            return x
