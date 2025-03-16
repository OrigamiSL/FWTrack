import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch import Tensor
from torch.nn import Identity
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
import copy

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPathAllocator:
    def __init__(self, max_drop_path_rate, stochastic_depth_decay = True):
        self.max_drop_path_rate = max_drop_path_rate
        self.stochastic_depth_decay = stochastic_depth_decay
        self.allocated = []
        self.allocating = []

    def __enter__(self):
        self.allocating = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.allocating) != 0:
            self.allocated.append(self.allocating)
        self.allocating = None
        if not self.stochastic_depth_decay:
            for depth_module in self.allocated:
                for module in depth_module:
                    if isinstance(module, DropPath):
                        module.drop_prob = self.max_drop_path_rate
        else:
            depth = self.get_depth()
            dpr = [x.item() for x in torch.linspace(0, self.max_drop_path_rate, depth)]
            assert len(dpr) == len(self.allocated)
            for drop_path_rate, depth_modules in zip(dpr, self.allocated):
                for module in depth_modules:
                    if isinstance(module, DropPath):
                        module.drop_prob = drop_path_rate

    def __len__(self):
        length = 0

        for depth_modules in self.allocated:
            length += len(depth_modules)

        return length

    def increase_depth(self):
        self.allocated.append(self.allocating)
        self.allocating = []

    def get_depth(self):
        return len(self.allocated)

    def allocate(self):
        if self.max_drop_path_rate == 0 or (self.stochastic_depth_decay and self.get_depth() == 0):
            drop_path_module = Identity()
        else:
            drop_path_module = DropPath()
        self.allocating.append(drop_path_module)
        return drop_path_module

    def get_all_allocated(self):
        allocated = []
        for depth_module in self.allocated:
            for module in depth_module:
                allocated.append(module)
        return allocated

class TargetQueryDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer, self).__init__()
        self.norm_1 = norm_layer(dim)

        self.self_attn1 = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)

        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm_3 = norm_layer(dim)
        self.norm_4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlpz = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_seq = norm_layer(dim)

        self.drop_path = drop_path


    def forward(self, query,  memoryxz,  query_pos, pos_xz, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                ):

        '''
            Args:
                query (torch.Tensor): (B, num_queries, C)
                memory (torch.Tensor): (B, L, C)
                query_pos (torch.Tensor): (1 or B, num_queries, C)
                memory_pos (torch.Tensor): (1 or B, L, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        q2 = self.norm_2_query(query) + query_pos       
        memory = memoryxz
        k2 = (self.norm_2_memory(memory) + pos_xz).permute(1, 0, 2)
        memory_in = memory.permute(1, 0, 2)
        query = query + self.drop_path(
            self.multihead_attn(query=q2, key=k2, value=memory_in, attn_mask=memory_mask,
                            key_padding_mask=memory_key_padding_mask)[0])
        query = query + self.drop_path(self.mlpz(self.norm_3(query)))

        query = self.norm_seq(query)

        return query
    

    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TargetQueryDecoderBlock(nn.Module):
    def __init__(self, dim, decoder_layers, num_layer,query_attention_layers = None,selfatt=False):
        super(TargetQueryDecoderBlock, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.num_layers = num_layer
        self.norm = nn.LayerNorm(dim)
        self.query_attention_layers = query_attention_layers

    
    def forward(self, 
                tgt, zx_feat, pos_embed_total,  query_pos: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_mask_onlyseq: Optional[Tensor] = None):

        output = tgt
        
        for i in range(len(self.layers)):
                # print('1')
            layer = self.layers[i]
            output = layer(output,zx_feat, query_pos, pos_embed_total,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
            
        output = self.norm(output)

        return output

def build_decoder(decoder_layer, drop_path, dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, z_size, x_size,two_xfeat = False,selfatt_seq = False):
    z_shape = [z_size, z_size]
    x_shape = [x_size, x_size]
    num_layers = decoder_layer
    
    query_attention_layers = None
    
    decoder_layers = []

    for _ in range(num_layers):
        decoder_layers.append(
            TargetQueryDecoderLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path.allocate()))
        drop_path.increase_depth()
   
    decoder = TargetQueryDecoderBlock(dim, decoder_layers, num_layers, query_attention_layers= query_attention_layers)
    return decoder

def get_sinusoid_encoding_table(n_position, d_hid, cls_token=False):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    # pos_embed = torch.FloatTensor(sinusoid_table).unsqueeze(0)
    pos_embed = sinusoid_table
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, d_hid]), pos_embed], axis=0)
    return pos_embed



class Pix2Track(nn.Module):
    def __init__(self, in_channel=64, feat_sz=20, feat_tz=10, range=2, pre_number=3,stride=16, encoder_layer=3, decoder_layer=3,
                 bins=400,num_heads=12, mlp_ratio=2, qkv_bias=True, drop_rate=0.0,attn_drop=0.0, drop_path=nn.Identity, number_template =2,
                 cfg = None):
        super(Pix2Track, self).__init__()

        self.bins = bins
        self.range = range
        self.word_embeddings = nn.Embedding(self.bins * self.range + 2, in_channel, padding_idx=self.bins * self.range, max_norm=1, norm_type=2.0)
        self.embeddings_norm = nn.LayerNorm(in_channel)
        self.embeddings_drop = nn.Dropout(drop_rate)

        self.pre_number = pre_number
        
        self.position_embeddings = nn.Embedding(
            4, in_channel)
       
        self.output_bias = torch.nn.Parameter(torch.zeros(self.bins * self.range + 2))

        self.encoder_layer = encoder_layer
        self.drop_path = drop_path
        self.tz = feat_tz * feat_tz
        self.sz = feat_sz * feat_sz   
        trunc_normal_(self.word_embeddings.weight, std=.02)
        self.total_size =  (self.sz + number_template*self.tz ) 
        total_size = self.sz + 2*self.tz
 
        self.pos_embed_total = nn.Parameter(torch.zeros(1, total_size, in_channel))
        pos_embed_total = get_sinusoid_encoding_table(total_size, self.pos_embed_total.shape[-1], cls_token=False)
        self.pos_embed_total.data.copy_(torch.from_numpy(pos_embed_total).float().unsqueeze(0)) 

        self.decoder = build_decoder(decoder_layer, self.drop_path, in_channel, num_heads,
                                     mlp_ratio, qkv_bias, drop_rate, attn_drop, feat_tz, feat_sz)
        

    def forward(self, zx_feat=None, wave_feat = None, wave_pos_embed = None):
        
        share_weight = self.word_embeddings.weight.T     

        bs = zx_feat.shape[0]

        origin_seq = 0.5 * torch.ones(bs, 4) * self.bins * self.range
        origin_seq = origin_seq.to(zx_feat.device).to(torch.int64)
        tgt = self.embeddings_drop(self.embeddings_norm(self.word_embeddings(origin_seq))).permute(1, 0, 2)

        query_embed = self.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)

        input_pos_embed = self.pos_embed_total[:, 0:self.total_size, :]
        if wave_feat != None:     
            zx_feat = torch.cat([zx_feat, wave_feat], dim = 1)
            input_pos_embed = torch.cat([input_pos_embed, wave_pos_embed], dim = 1)

        decoder_feat = self.decoder(tgt, zx_feat, input_pos_embed,
                                    query_embed,
                                    tgt_mask= None)
        
        out = torch.matmul(decoder_feat.transpose(0, 1), share_weight)
   
        out +=  self.output_bias.expand(out.shape[0], out.shape[1], out.shape[2])

        outfeat = out.permute(1,0,2)

        out = out.softmax(-1)
        feat_for_test = out

        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        seqs_output = extra_seq
        values = value
        
        output = {'feat':outfeat,'seqs': seqs_output, 'class': values, "state": "val/test",
                    'confidence':value, 'feat_for_test':feat_for_test}
                                
        return output

def build_pix_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE
    x_number_patches = hidden_dim * (cfg.DATA.SEARCH.SIZE // stride)**2
    
    if cfg.MODEL.HEAD.TYPE == "PIX":
        in_channel = hidden_dim
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        feat_tz = int(cfg.DATA.TEMPLATE.SIZE / stride)
        decoder_layer = cfg.MODEL.DECODER_LAYER
        encoder_layer = cfg.MODEL.ENCODER_LAYER
        bins = cfg.MODEL.BINS
        num_heads = cfg.MODEL.NUM_HEADS
        mlp_ratio = cfg.MODEL.MLP_RATIO
        qkv_bias = cfg.MODEL.QKV_BIAS
        drop_rate = cfg.MODEL.DROP_RATE
        attn_drop = cfg.MODEL.ATTN_DROP
        drop_path = cfg.MODEL.DROP_PATH
        drop_path_allocator = DropPathAllocator(drop_path)
        range = cfg.MODEL.RANGE
        pre_number = cfg.DATA.SEARCH.NUMBER -1
        number_template = cfg.DATA.TEMPLATE.NUMBER
        pix_head = Pix2Track(in_channel=in_channel, feat_sz=feat_sz, feat_tz=feat_tz, range=range,pre_number=pre_number,
                             stride=stride, encoder_layer=encoder_layer, decoder_layer=decoder_layer, bins=bins,
                             num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                             attn_drop=attn_drop, drop_path=drop_path_allocator, number_template = number_template,
                             cfg = cfg)
        return pix_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
