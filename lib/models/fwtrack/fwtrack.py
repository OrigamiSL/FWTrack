"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.head import build_pix_head
from lib.models.fwtrack.vit import vit_base_patch16_224, vit_large_patch16_224
import time

class FWTrack(nn.Module):
    """ This is the base class for FWTrack """

    def __init__(self, transformer_S, pix_head, hidden_dim, wave_processor):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
        """
        super().__init__()    
        
        self.backbone_S = transformer_S
        
        self.pix_head = pix_head

        self.wave_processor = wave_processor
    
    def forward(self, 
                zx_feat = None,     
                wave_index = None,
                rest_index = None
                ):  

        feat_S = self.backbone_S(zx_feat = zx_feat, wave_index = wave_index, rest_index = rest_index)
        
        out = self.forward_head(feat_S)
             
        return out

    def forward_head(self, feat_S):
        
        output_dict = self.pix_head(feat_S)
        return output_dict
        

def build_fwtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    pretrained_path = os.path.join('/home/lhg/work/fxy_ar1/AR_one_sequence/pretrained_models/')

    if cfg.MODEL.PRETRAIN_FILE and ('FWTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone_S = vit_base_patch16_224(cfg, pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone_S.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone_S.finetune_track(cfg=cfg, patch_start_index=patch_start_index, if_smooth = True)
    pix_head = build_pix_head(cfg, hidden_dim)
    wave_processor = None
    model = FWTrack(
        backbone_S,
        pix_head,
        hidden_dim,
        wave_processor)
    
    if cfg.MODEL.PRETRAIN_PTH != "":
        load_from = cfg.MODEL.PRETRAIN_PTH
        checkpoint = torch.load(load_from, map_location="cpu")
        load_net_dict = checkpoint["net"]
        missing_keys, unexpected_keys = model.load_state_dict(load_net_dict, strict=False)

        print('Load pretrained model from: ' + load_from)
    if 'FWTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
