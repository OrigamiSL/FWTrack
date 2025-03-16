from . import BaseActor

import torch
import math
import numpy as np
import lib.train.data.bounding_box_utils as bbutils

from lib.models.fwtrack.modwt_mra import get_all_smooth_kernel,\
get_modwt_mra_level, get_modwt2d_level,  modwt2d_v_w_level, get_ker_level, get_mra_kernel_weight
from lib.models.fwtrack.modwt_mra_t import get_ker_level_t, get_all_smooth_kernel_t, modwt2d_t_level
import time
import lib.train.data.processing_utils as prutils
from torchvision.transforms import Resize
import cv2

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


# angle cost
def SIoU_loss(test1, test2, theta=4):
    eps = 1e-7
    cx_pred = (test1[:, 0] + test1[:, 2]) / 2
    cy_pred = (test1[:, 1] + test1[:, 3]) / 2
    cx_gt = (test2[:, 0] + test2[:, 2]) / 2
    cy_gt = (test2[:, 1] + test2[:, 3]) / 2

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(test1[:, 0], test2[:, 0])
    xmax = torch.max(test1[:, 2], test2[:, 2])
    ymin = torch.min(test1[:, 1], test2[:, 1])
    ymax = torch.max(test1[:, 3], test2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = test1[:, 2] - test1[:, 0]
    h_pred = test1[:, 3] - test1[:, 1]
    w_gt = test2[:, 2] - test2[:, 0]
    h_gt = test2[:, 3] - test2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    lt = torch.max(test1[..., :2], test2[..., :2])  # [B, rows, 2]
    rb = torch.min(test1[..., 2:], test2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (test1[..., 2] - test1[..., 0]) * (
            test1[..., 3] - test1[..., 1])
    area2 = (test2[..., 2] - test2[..., 0]) * (
            test2[..., 3] - test2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou

class FWTrackActor(BaseActor):
    """ Actor for training FWTrack models """

    def __init__(self, net, objective, loss_weight, settings, bins, search_size, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.bins = bins
        self.range = self.cfg.MODEL.RANGE
        self.search_size = search_size
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.focal = None
        self.loss_weight['KL'] = 100
        self.loss_weight['focal'] = 2
        self.pre_num = self.cfg.DATA.SEARCH.NUMBER -1
        self.device = torch.device('cuda:0')
        self.level_s = self.cfg.DATA.WAVELETS_LEVEL_S
        
        self.modwt_kernel = get_ker_level(level = self.level_s, type_k = 'smooth')
        
        mra_kernel = get_all_smooth_kernel(level = self.level_s)
        self.mra_kernel_weight_x = get_mra_kernel_weight(mra_kernel, self.cfg.DATA.SEARCH.SIZE//2, device = self.device, level = self.level_s)
        
        mean = torch.tensor(self.cfg.DATA.MEAN).view((1, 3, 1, 1)).to(self.device)
        std = torch.tensor(self.cfg.DATA.STD).view((1, 3, 1, 1)).to(self.device)

        self.mean = mean.expand((4+self.cfg.DATA.TEMPLATE.NUMBER) * (self.level_s + 1) * self.cfg.TRAIN.BATCH_SIZE, -1, self.cfg.DATA.SEARCH.SIZE//2, self.cfg.DATA.SEARCH.SIZE//2)
        self.std = std.expand((4+self.cfg.DATA.TEMPLATE.NUMBER) * (self.level_s + 1) * self.cfg.TRAIN.BATCH_SIZE, -1, self.cfg.DATA.SEARCH.SIZE//2, self.cfg.DATA.SEARCH.SIZE//2)

        self.weight_list = []
        for i in range(self.level_s):
            weight_s = torch.FloatTensor(self.modwt_kernel[i]).unsqueeze(0).unsqueeze(0)
            input_weight_s = torch.nn.Parameter(weight_s, requires_grad = False).to(self.device)
            self.weight_list.append(input_weight_s)

        self.mra_kernel_t = get_all_smooth_kernel_t()

        self.resize = Resize([self.cfg.DATA.WAVELETS_SEARCH_CROP, self.cfg.DATA.WAVELETS_SEARCH_CROP])

        # self.wave_norm = torch.nn.LayerNorm(self.cfg.DATA.SEARCH_CROP**2, elementwise_affine = False)
        # self.zeros = torch.zeros(self.cfg.TRAIN.BATCH_SIZE).to(self.device)
        # self.test_tensor = torch.zeros((16, 32, 512, 512, 3)).to(self.device)
        # self.zero_tensor = torch.zeros((self.cfg.TRAIN.BATCH_SIZE, 16*3, self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE)).to(self.device)

        self.level_t = cfg.DATA.WAVELETS_LEVEL_T
        self.level_hw = cfg.DATA.WAVELETS_LEVEL_HW
        modwt_kernel = get_ker_level(self.level_hw, 'smooth')
        modwt_kernel_diff = get_ker_level(self.level_hw, 'diff')
        self.g_k = modwt_kernel
        self.h_k = modwt_kernel_diff
        weight_list = []
        for i in range(self.level_hw):
            weight_i = torch.FloatTensor(modwt_kernel[i]).unsqueeze(0).unsqueeze(0)
            input_weight_i = torch.nn.Parameter(weight_i, requires_grad = False).to(self.device)
            weight_list.append(input_weight_i)
        self.g_weight = weight_list

        weight_list_diff = []
        for i in range(self.level_hw):
            weight_i = torch.FloatTensor(modwt_kernel_diff[i]).unsqueeze(0).unsqueeze(0)
            input_weight_i = torch.nn.Parameter(weight_i, requires_grad = False).to(self.device)
            weight_list_diff.append(input_weight_i)
        self.h_weight = weight_list_diff

        self.g_k_t = get_ker_level_t(self.level_t, 'smooth')
        self.h_k_t = get_ker_level_t(self.level_t, 'diff')
        self.g_weight_t = []
        self.h_weight_t = []

        for i in range(self.level_t):
            weight_i_g = torch.FloatTensor(self.g_k_t[i]).unsqueeze(0).unsqueeze(0)
            input_weight_i_g = torch.nn.Parameter(weight_i_g, requires_grad = False).to(self.device)
            self.g_weight_t.append(input_weight_i_g)

            weight_i_h = torch.FloatTensor(self.h_k_t[i]).unsqueeze(0).unsqueeze(0)
            input_weight_i_h = torch.nn.Parameter(weight_i_h, requires_grad = False).to(self.device)
            self.h_weight_t.append(input_weight_i_h)
        
    def __call__(self, data, flag = 0, seq_feat = None, x_feat = None):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """     
        out_dict = self.forward_pass(data, flag)

        loss, status = self.compute_losses(out_dict, data,flag = 0)
        
        return loss, status
    
    def get_S_D(self, image):
        B, C, H, W = image.shape
        
        img_wavelets = get_modwt2d_level(image, self.modwt_kernel, self.weight_list, self.device, level = self.level_s)
        img_mra = get_modwt_mra_level(img_wavelets, self.mra_kernel_weight_x, self.device, level = self.level_s)      
        img_mra.insert(0, image)

        img_all = torch.stack(img_mra, dim = 0) # 4 * 64 * 
        img_all = img_all.view((self.level_s+1)*B, C, H, W)
        
        img_tensor_norm = ((img_all) - self.mean) / self.std

        return img_tensor_norm
    
    def wavelets_t_diff(self, image, wave_norm=None, use_hw_wavelets = False, only_w3 = True):
        #  [512, 32, 512, 3, 8])
        H, B, W, C, t = image.shape
        # t, B, H, W, C = image.shape #16, 32, 512, 512, 3
        image = image.permute(4, 1, 3, 0, 2).contiguous().view(-1, C, H, W) #16, 32, 3, 512, 512 - >
        if use_hw_wavelets: 
            image_v_w_list =  modwt2d_v_w_level(image, self.g_k,self.h_k, self.g_weight, self.h_weight,self.device, self.level_hw, only_w3 = only_w3)
            v_w_list_len = len(image_v_w_list)
            image_v_w = torch.stack(image_v_w_list, dim = 0) # 8 *16, 32, 3, 512, 512
        else:
            image_v_w = image.unsqueeze(0)
            v_w_list_len = 1

        image_v_w = image_v_w.transpose(0, 1) # 16, [8, 32, 3, 512, 512]
        if wave_norm == None:
            image_v_w_output = image_v_w.clone()

        image = image_v_w.contiguous().view(t, -1).permute(1, 0) #N, t
        N, t = image.shape
        
        img_wavelets = modwt2d_t_level(image, self.g_k_t, self.h_k_t, self.g_weight_t, self.h_weight_t, self.device, self.level_t) # 2, N, t

        img_wavelets_last = img_wavelets[:, :, -1]

        result = img_wavelets_last.contiguous().view(self.level_t, v_w_list_len, B, C, H, W)
        if wave_norm != None:
            result = result.permute(2, 0, 1, 3, 4, 5)
            result = result.contiguous().view(B, v_w_list_len*self.level_t*C, -1)
            result = wave_norm(result).view(B, v_w_list_len*self.level_t*C, H, W)
  
            return result
        else:
            return image_v_w_output, result
     
    def crop_jitter(self, img, jitter, hw):
    
        img = prutils.sample_target_tensor(img, jitter, self.cfg.DATA.SEARCH.FACTOR, self.resize, hw, self.zeros)
        
        return img

    def forward_pass(self, data, flag):
        
        data_template = data['template_images'].squeeze(1).view(-1, *data['template_images'].shape[3:])

        x = data['search_images'][-1].view(-1, *data['search_images'].shape[2:]).squeeze(0) # 32, 3, 288, 288

        B, C, H, W = x.shape
        x = x.contiguous().view(B, C, 2, H//2, 2, W//2).permute(2, 4, 0, 1, 3, 5)\
            .contiguous().view(4*B, C, H//2, W//2)
        zx_feat = torch.cat([x, data_template], dim = 0)

        zx_feat = self.get_S_D(zx_feat)
        
        true_box = data['search_anno'].squeeze(0).squeeze(0)

        true_box[:, 2] = true_box[:, 0] + true_box[:, 2]
        true_box[:, 3] = true_box[:, 1] + true_box[:, 3] 

        magic_num = (self.range - 1) * 0.5
        true_box = true_box.clamp(min=(-1*magic_num), max=(1+magic_num))

        seq_true = (true_box + magic_num) * (self.bins - 1)
        seq_true = seq_true.int().to(x)

        seq_output = torch.tensor(seq_true)
             
        data['seq_output'] = seq_output
      
        out_dict = self.net(zx_feat = zx_feat,  
                            wave_index = None)
        
        return out_dict
                
    def compute_losses(self, pred_dict, gt_dict, return_status=True, flag = 0, seq_feat = None, x_feat = None):
        
        bins = self.bins
        magic_num = (self.range - 1) * 0.5
        seq_output = gt_dict['seq_output']
        pred_feat = pred_dict["feat"]
        

        if self.focal == None:
            weight = torch.ones(bins*self.range+2) * 1
            weight[bins*self.range+1] = 0.1
            weight[bins*self.range] = 0.1
            weight.to(pred_feat)
            self.klloss = torch.nn.KLDivLoss(reduction='none').to(pred_feat)
            self.focal = torch.nn.CrossEntropyLoss(weight=weight, size_average=True).to(pred_feat)
       
        pred = pred_feat.permute(1, 0, 2).reshape(-1, bins*2+2)
       
        target = seq_output.reshape(-1).to(torch.int64)

        varifocal_loss = self.focal(pred, target)
        
        pred = pred_feat[0:4, :, 0:bins*self.range] 
        
        target = seq_output[:, 0:4].to(pred_feat)
        
        out = pred.softmax(-1).to(pred)
        
        mul = torch.range((-1*magic_num), (1+magic_num), 2/(self.bins*self.range))[:-1].to(pred)
        
        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)

        target = target / (bins - 1) - magic_num

        extra_seq = ans
        extra_seq = extra_seq.to(pred)
        sious, iou = SIoU_loss(extra_seq, target, 4)
        sious = sious.mean()
        siou_loss = sious
  
        l1_loss = self.objective['l1'](extra_seq, target)

        loss = self.loss_weight['giou'] * siou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * varifocal_loss 
        
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            
            if flag == 0:  
                status = {"Loss/total": loss.item(),
                        "Loss/giou": siou_loss.item(),
                        "Loss/l1": l1_loss.item(),
                        "Loss/location": varifocal_loss.item(),
                        "IoU": mean_iou.item(),}
                    
            return loss, status
        
        else:
            return loss
