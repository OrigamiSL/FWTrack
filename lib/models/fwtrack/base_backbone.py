from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed

import time

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        self.x_patch_number_hw = 18
        self.z_patch_number_hw = 9

        self.sep_type = 3

    def finetune_track(self, cfg, patch_start_index=1, if_smooth = False):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        self.template_number = cfg.DATA.TEMPLATE.NUMBER
        
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                print('name', name)
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim,stride_size= 16)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']
        
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = search_size

        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.x_patch_number_hw = new_P_H
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)

        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.z_patch_number_hw = new_P_H
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        
        self.pos_embed_z_1 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_z_2 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_z_3 = nn.Parameter(template_patch_pos_embed) #1, 81, 256
        self.pos_embed_z_4 = nn.Parameter(template_patch_pos_embed)


        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)


    def forward_features(self, zx_feat, wave_index = None, rest_index = None, if_smooth=False):
        
        zx_feat = self.patch_embed(zx_feat, if_smooth = True)
        B_all, C, H, W = zx_feat.shape
        B = B_all // (4 + self.template_number)

        x = zx_feat[:B * 4, ...]
        z_all = zx_feat[B * 4:, ...]

        x = x.contiguous().view(2, 2, B, C, H, W).permute(2, 3, 0, 4, 1, 5)\
        .contiguous().view(B, C, 2*H, 2*W)
        x += self.pos_embed_x

        for i in range(self.template_number):
            z_all[i*B:(i+1)*B, :, :, :] += eval('self.pos_embed_z_'+str(i+1))
        
        z_all = z_all.flatten(2).transpose(1, 2)
        z_all = z_all.contiguous().view(self.template_number, B, H*W, C).transpose(0, 1)\
        .contiguous().view(B, self.template_number*H*W, C)
            
        x = x.flatten(2).transpose(1, 2) # B, N, C

        x_patch_list = torch.cat([x, z_all], dim = 1)

        x_patch_list = self.pos_drop(x_patch_list)

        for i, blk in enumerate(self.blocks):

            x_patch_list = blk(x_patch_list, self.template_number, wave_index, rest_index)


        xz = self.norm(x_patch_list)  

        return xz
    
    def forward(self, zx_feat, wave_index=None, rest_index = None, if_smooth=False, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x = self.forward_features(zx_feat, wave_index, rest_index, if_smooth)

        return x
