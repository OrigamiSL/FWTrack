import math

from lib.models.fwtrack import build_fwtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.train.data.processing_utils import sample_target, sample_target_4corner
import cv2
import os
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.models.fwtrack.modwt_mra import get_all_smooth_kernel,get_modwt2d_level, get_modwt_mra_level,get_mra_kernel_weight,get_ker_level
from lib.models.fwtrack.modwt_mra_t import modwt2d_t_level, get_ker_level_t
import random
import lib.train.data.processing_utils as prutils
from torchvision.transforms import Resize
import time

class FWTrack(BaseTracker):
    def __init__(self, params, dataset_name=None, test_checkpoint=None,net_for_test=None,update_intervals=None,update_threshold=None,hanning_size=None,\
                 pre_seq_number=None,std_weight=None, smooth_type=None, alpha=None, beta=None, double_dayu=None, smooth_thre=None):
        super(FWTrack, self).__init__(params)
        if net_for_test == None:
            print('build new network for test')
            network = build_fwtrack(params.cfg, training=False)
            self.network = network.cuda()
            self.network.eval()
        else:
            print('use trained network for test')
            network = net_for_test
        
        if test_checkpoint == None or test_checkpoint == 'test_net':
            print('dont use trained checkpoint')
        else:
            print('load_checkpoint_path',test_checkpoint)
            network.load_state_dict(torch.load(test_checkpoint, map_location='cpu')['net'], strict=False)

        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.use_frame_number = self.cfg.DATA.SEARCH.NUMBER
        
        self.preprocessor = Preprocessor()
        self.state = None
        self.range = self.cfg.MODEL.RANGE
        self.num_template = self.cfg.TEST.TEMPLATE_NUMBER
        print('template number',self.num_template)

        self.hanning_simple = torch.tensor(np.hanning(2*self.bins+2)).unsqueeze(0).expand(4, 2*self.bins+2).cuda()

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.debug = False
        self.use_visdom = False
        self.frame_id = 0
        self.save_all_boxes = params.save_all_boxes
        self.dataset_name = dataset_name

        self.z_dict1 = {}
        DATASET_NAME = dataset_name.upper()

        if update_intervals == None:
            if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
                self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            else:
                print('use_default_inter')
                self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        else:
            self.update_intervals = int(update_intervals)
        print("Update interval is: ", self.update_intervals)
        
        if update_threshold == None:
            if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
                self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
            else:
                print('use_default_thre')
                self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        else:
            self.update_threshold = float(update_threshold)
        print("Update threshold is: ", self.update_threshold)   

        self.current_frame = 1
        self.multi_weight = 0.9
        print('self.multi_weight',self.multi_weight)

        if hasattr(self.cfg.TEST.HANNING_SIZE,DATASET_NAME):
            self.hanning_size = self.cfg.TEST.HANNING_SIZE[DATASET_NAME]
        else:
            print('use_default_smooth_thre')
            self.hanning_size = self.cfg.TEST.HANNING_SIZE.DEFAULT

        print('self.hanning_size',self.hanning_size)

        if pre_seq_number  == None:
            if dataset_name == 'uav':
                self.pre_seq_number = 5
            else:
                self.pre_seq_number = self.cfg.DATA.SEARCH.USE_PREDICT # 7
        else:
            self.pre_seq_number = int(pre_seq_number)
        print('self.pre_seq_number',self.pre_seq_number)

        if hasattr(self.cfg.TEST.STD_WEIGHT,DATASET_NAME):
            self.std_weight = self.cfg.TEST.STD_WEIGHT[DATASET_NAME]
        else:
            print('use_default_STD_WEIGHT')
            self.std_weight = self.cfg.TEST.STD_WEIGHT.DEFAULT

        print('self.std_weight',self.std_weight)
        
        if alpha != None:
            self.alpha = float(alpha)
        else:
            self.alpha = self.cfg.TEST.ALPHA
        print('alpha',self.alpha)

        if hasattr(self.cfg.TEST.BETA,DATASET_NAME):
            self.beta = self.cfg.TEST.BETA[DATASET_NAME]
        else:
            print('use_default_beta')
            self.beta = self.cfg.TEST.BETA.DEFAULT
        print('beta',self.beta)

        if hasattr(self.cfg.TEST.SMOOTH_THRESHOLD, DATASET_NAME):
            self.smooth_thre = self.cfg.TEST.SMOOTH_THRESHOLD[DATASET_NAME]
        else:
            print('use_default_smooth_thre')
            self.smooth_thre = self.cfg.TEST.SMOOTH_THRESHOLD.DEFAULT
        print('smooth_thre', self.smooth_thre)

        self.device = torch.device('cuda:0')
        self.level_s = self.cfg.DATA.WAVELETS_LEVEL_S
        self.modwt_kernel = get_ker_level(level = self.level_s, type_k = 'smooth')
        self.mra_kernel = get_all_smooth_kernel()

        mra_kernel = get_all_smooth_kernel(level = self.level_s)
        self.mra_kernel_weight_x = get_mra_kernel_weight(mra_kernel, self.cfg.DATA.TEMPLATE.SIZE, device = self.device, level = self.level_s)
        
        self.weight_list = []
        for i in range(self.level_s):
            weight_s = torch.FloatTensor(self.modwt_kernel[i]).unsqueeze(0).unsqueeze(0)
            input_weight_s = torch.nn.Parameter(weight_s, requires_grad = False).to(self.device)
            self.weight_list.append(input_weight_s)

        self.level_t = self.cfg.DATA.WAVELETS_LEVEL_T

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

        # self.wave_norm = torch.nn.LayerNorm(self.cfg.DATA.SEARCH_CROP * self.cfg.DATA.SEARCH_CROP, elementwise_affine = False)
        self.resize = Resize([self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE])
        self.resize_z = Resize([self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE])

        self.wave_thre = 0.75

        print('self.wave_thre', self.wave_thre )
        self.x_index = (self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE)**2
        self.z_index = (self.cfg.DATA.TEMPLATE.SIZE // self.cfg.MODEL.BACKBONE.STRIDE)**2

        template_index = [ i for i in range(self.x_index, self.x_index + self.z_index * self.cfg.DATA.TEMPLATE.NUMBER)]
        self.init_template_index = torch.Tensor(template_index).cuda().int()

        template_index = [ i for i in range(self.x_index, self.x_index + self.z_index)]
        self.static_template_index = torch.Tensor(template_index).cuda().int()

        self.use_past_wave = True
        print('use_past_wave', self.use_past_wave)

        self.corner_test = False
        print('use_corner_test', self.corner_test)

        mean = torch.tensor(self.cfg.DATA.MEAN).view((1, 3, 1, 1)).to(self.device)
        std = torch.tensor(self.cfg.DATA.STD).view((1, 3, 1, 1)).to(self.device)
        self.zeros = torch.zeros(1).cuda()
        
        if self.corner_test:
            self.mean = mean.expand((self.level_s+1)*(4 + self.num_template)*5 , -1, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE)
            self.std = std.expand((self.level_s+1)*(4 + self.num_template)*5 , -1, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE)
        else:
            self.mean = mean.expand((self.level_s+1)*(4 + self.num_template) , -1, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE)
            self.std = std.expand((self.level_s+1)*(4 + self.num_template) , -1, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE)

    def get_S_D(self, x, z):

        x = torch.tensor(x).cuda().float()
        z = torch.tensor(z).cuda().float()
        
        x = x.contiguous().view(1, 2, self.cfg.DATA.TEMPLATE.SIZE, 2, self.cfg.DATA.TEMPLATE.SIZE, 3).permute(1, 3, 0, 2, 4, 5)\
        .contiguous().view(4, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE, 3)

        image = torch.cat([x, z], dim = 0)
        image = image.permute(0,3,1,2)
        
        B, C, H, W = image.shape
        img_wavelets = get_modwt2d_level(image, self.modwt_kernel, self.weight_list, self.device, level = self.level_s)
        img_mra = get_modwt_mra_level(img_wavelets, self.mra_kernel_weight_x, self.device, level = self.level_s)      
        img_mra.insert(0, image)
       
        img_all = torch.stack(img_mra, dim = 0) # 4 * 64 * 
        img_all = img_all.view((self.level_s+1)*B, C, H, W)
        
        img_tensor_norm = ((img_all/255) - self.mean) / self.std

        return img_tensor_norm

    def wavelets_t_diff(self, image_list, if_draw = False):
        # image = np.stack(image_list, axis = -1) # 512, 512, 3, 8
        # image = torch.tensor(image).cuda().float().unsqueeze(0) # 1, 512, 512, 3, 8
        image = image_list.unsqueeze(0)

        B, H, W, C, t = image.shape
        image = image.contiguous().view(-1, t)
        img_wavelets = modwt2d_t_level(image, self.g_k_t, self.h_k_t, self.g_weight_t, self.h_weight_t, self.device, self.level_t)
        img_wavelets_last = img_wavelets[:, :, -1]
        result = img_wavelets_last.contiguous().view(self.level_t, B, H, W, C)
       
        last_result = torch.sum(result, dim = 0)
        last_result = last_result.squeeze(0)
        
        return last_result
 
    def crop_jitter(self, img, jitter, hw):

        img = prutils.sample_target_tensor(img, jitter, self.cfg.DATA.SEARCH.FACTOR, self.resize, hw)
        
        return img
    
    def get_wave_patch_index(self, img, jitter, hw, thre, id = None, if_update_z = False):
        
        if if_update_z == False:
            wave_index, rest_index = prutils.sample_wave_index(img, jitter, self.cfg.DATA.SEARCH.FACTOR, self.resize, hw, 
                                        zeros_input= self.zeros, thre= thre, id = id)
        else:
            wave_index, rest_index = prutils.sample_wave_index(img, jitter, self.cfg.DATA.TEMPLATE.FACTOR, self.resize_z, hw, 
                                        zeros_input= self.zeros, thre= thre, id = id)
            wave_index += (self.x_index + self.z_index)
            rest_index += (self.x_index + self.z_index)
        
        return wave_index, rest_index

    def initialize(self, image, info: dict):
        H, W, _ = image.shape
        self.input_hw = torch.Tensor((H,W)).cuda().unsqueeze(0)
        # forward the template once

        z_patch_arr, _, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)#output_sz=self.params.template_size
        seq_len = self.cfg.DATA.SEARCH.NUMBER
        
        if self.use_past_wave:
            self.past_img_list = []
            init_crop = cv2.resize(image, (self.cfg.DATA.SEARCH_CROP, self.cfg.DATA.SEARCH_CROP))
            init_crop = torch.tensor(init_crop).cuda().float()
            for i in range(seq_len):
                self.past_img_list.append(init_crop)
            self.past_img_list = torch.stack(self.past_img_list, dim = -1)
            
        with torch.no_grad():
            self.template_list = [z_patch_arr] * self.num_template

        self.box_mask_z = None
        # save states
        self.state = info['init_bbox']

        self.wave_index_z = None
        self.rest_index_z = None

    def track(self, image, current_frame, info: dict = None):
        
        magic_num = (self.range - 1) * 0.5
        H, W, _ = image.shape
        
        self.frame_id += 1
        if self.corner_test:
            x_patch_arr, resize_factor, jitter_list = sample_target_4corner(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  
            for k in range(x_patch_arr.shape[0]):
                cv2.imwrite(str(k)+'.jpg', x_patch_arr[k])     
        else:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
            
        z_list = np.stack(self.template_list, axis = 0)
        input_feat = self.get_S_D(x_patch_arr, z_list)

        if self.use_past_wave:
            wave_image = cv2.resize(image, (self.cfg.DATA.SEARCH_CROP, self.cfg.DATA.SEARCH_CROP))
            wave_image = torch.tensor(wave_image).cuda().float()

            self.past_img_list[:, :, :, :-1] = self.past_img_list[:, :, :, 1:].clone()
            self.past_img_list[:, :, :, -1] = wave_image

            jitter = torch.tensor(self.state).cuda().unsqueeze(0)
            wavelet_feat = self.wavelets_t_diff(self.past_img_list)

            wave_index, rest_index = self.get_wave_patch_index(wavelet_feat, jitter, 
                        self.input_hw, thre = self.wave_thre, id = str(self.current_frame),
                        if_update_z = False)

            if self.wave_index_z != None:
                wave_index = torch.cat([wave_index, self.static_template_index, self.wave_index_z], dim = 0)
                rest_index = torch.cat([rest_index, self.rest_index_z], dim = 0)
            else:
                wave_index = torch.cat([wave_index, self.init_template_index], dim = 0)
          
        else:
            wave_index = None
            rest_index = None

        with torch.no_grad():  
            out_dict = self.network.forward(
                zx_feat = input_feat, wave_index = wave_index, rest_index = rest_index)

        feat_for_test = out_dict['feat_for_test']

        if self.corner_test:
            conf = out_dict['confidence'].sum(dim = -2).squeeze(-1)
            index = torch.argmax(conf, dim = 0)
            conf = conf[index].item() * 10
            feat_for_test = feat_for_test[index, :, :]
        else:
            conf = out_dict['confidence'].sum().item()*10

        _, extra_seq = feat_for_test.topk(dim = -1, k= 1)[0], feat_for_test.topk(dim = -1, k= 1)[1]
        
        pred_boxes = extra_seq[:, 0:4] / (self.bins - 1) - magic_num
        
        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)

        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_boxes[2]/2
        pred_new[1] = pred_boxes[1] + pred_boxes[3]/2

        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        if self.corner_test:
            self.state = clip_box(self.map_box_back_corner_test(pred_boxes, resize_factor, index, jitter_list), H, W, margin=10)
        else:
            self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:

            if (self.frame_id % self.update_intervals == 0) and (conf > self.update_threshold):
                self.wave_index_z, self.rest_index_z = self.get_wave_patch_index(wavelet_feat, jitter, 
                        self.input_hw, thre = self.wave_thre, id = str(self.current_frame),
                        if_update_z = True)
                z_patch_arr, _, _ = sample_target(image, self.state, self.params.template_factor,
                                            output_sz=self.params.template_size)
                self.template_list.append(z_patch_arr)
                self.template_list.pop(1)
            
        self.current_frame += 1

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        #cx_real = cx + cx_prev
        #cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
    
    def map_box_back_corner_test(self, pred_box: list, resize_factor: float, index, jitter_list):
        index = index.item()

        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor

        cx_real = cx + (cx_prev + jitter_list[index][0] - half_side)
        cy_real = cy + (cy_prev + jitter_list[index][1]- half_side)
        #cx_real = cx + cx_prev
        #cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return FWTrack
