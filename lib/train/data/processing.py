import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import lib.train.data.transforms as tfm
import random
import cv2
import numpy as np
from torchvision.transforms import Resize
import time
from lib.models.fwtrack.modwt_mra_t import get_modwt2d_t
import multiprocessing
import matplotlib.pyplot as plt
 
def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,if_seq,cfg,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.if_seq = if_seq
        self.resize = Resize([512,512])
        self.cfg = cfg

        # self.device = 'cpu'
        self.device =  torch.device('cuda:0')

        self.resize = Resize([self.cfg.DATA.SEARCH.SIZE, self.cfg.DATA.SEARCH.SIZE])

        # self.wave_norm = torch.nn.LayerNorm(cfg.DATA.SEARCH_CROP*cfg.DATA.SEARCH_CROP, elementwise_affine = False)
        self.zeros = torch.zeros(1).to(self.device)

        # self.wave_list = spe_db2()

        self.hw_zeros = [torch.zeros(2)]
        #self.crop_img_zeros = [torch.zeros((cfg.DATA.SEARCH_CROP, cfg.DATA.SEARCH_CROP, 3)) for i in range(self.cfg.DATA.SEARCH.NUMBER)]
        self.id = 0
        # self.return_DMD_img = np.zeros([self.cfg.DATA.SEARCH_CROP, self.cfg.DATA.SEARCH_CROP, 3])

        # self.dmd_zeros = np.zeros(())

    
    def _get_jittered_box(self, box, mode,box_true=None):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        # 68%  0.779 - 1.28
        # 95%  0.607- 1.65
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)


    def get_S_D_t(self, image, wave_norm):
        image = image.to(self.device)
        t, B, H, W, C = image.shape #16, 32, 512, 512, 3
        image = image.contiguous().view(t, -1).permute(1, 0) 
        #t, -1
        N, t = image.shape
        
        img_wavelets,im1,im2 = get_modwt2d_t(image, self.modwt_kernel_t, self.weight_list_t, self.device)

        w1 = img_wavelets.permute(1, 0).view(t, B, H, W, C)
    
        w2 = im1.permute(1, 0).view(t, B, H, W, C)
       
        w3 = im2.permute(1, 0).view(t, B, H, W, C)
        
        w1 = w1[-1, :, :, :, :].permute(0, 3, 1, 2) # B, C, H, W
        w2 = w2[-1, :, :, :, :].permute(0, 3, 1, 2)
        w3 = w3[-1, :, :, :, :].permute(0, 3, 1, 2)

        result = torch.cat([w1, w2, w3], dim = 1)
        result = result.view(B, 3*C, -1)
        result = wave_norm(result).view(B,3*C, H, W)
        
        return result
    def crop_jitter(self,img, jitter, hw):

        # img = prutils.sample_target_tensor(img, jitter, self.cfg.DATA.SEARCH.FACTOR, self.resize, hw, self.zeros)
        img = prutils.sample_target_single(img, jitter, self.cfg.DATA.SEARCH.FACTOR, self.resize, hw, self.zeros)
      
        return img

    def __call__(self, data: TensorDict, use_search_frames: int, choose_multi: bool, use_DMD: bool):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        # print('use_search_frames', use_search_frames)
        if self.if_seq:
            if self.transform['joint'] is not None:

                data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
                
                data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                    image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
                

            for s in ['template', 'search']:
            
                assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                    "In pair mode, num train/test frames must be 1"

                if s == 'template':
                    jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                    # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                    w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                    crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                    if (crop_sz < 1).any():
                        
                        data['valid'] = False
                        # print("Too small box is found. Replace it with new data.")
                        return data
                    # Crop image region centered at jittered_anno box and get the attention mask
                    crops, boxes, att_mask, mask_crops,_ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                    data[s + '_anno'], self.search_area_factor[s],
                                                                                    self.output_sz[s], masks=data[s + '_masks'])
                    # Apply transforms
                    # print('tenpl')
                    data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                        image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                    # print('template_len', len(data[s + '_images']))
                else:
                    current_anno = data[s + '_anno'][-1]
                    current_frame = data[s + '_images'][-1]
                                       
                    if choose_multi == True:
                        if use_DMD :
                            init_search = data[s + '_images']
                            H, W, C = data[s + '_images'][0].shape
                            data['hw'] = [torch.Tensor((H,W))]
                                
                            search_1 = np.concatenate(init_search, axis = -1)
                            search_resize = cv2.resize(search_1,(self.cfg.DATA.SEARCH_CROP,self.cfg.DATA.SEARCH_CROP))
                            
                            search_resize = search_resize.reshape(self.cfg.DATA.SEARCH_CROP,self.cfg.DATA.SEARCH_CROP, use_search_frames, 3)
                            search_resize = search_resize.transpose((0, 1, 3, 2))

                            resize_search = torch.from_numpy(search_resize.copy()).float()  
                            data['crop_image'] = resize_search[:,:,:,-self.cfg.DATA.SEARCH.NUMBER:]
                            data['use_multi_frame'] = torch.ones(1)
                            
                            int_img = search_resize[:,:,:,-1].copy()
                            
                            init_wh = [H, W]
                            
                            current_dmd, x1_rand, x2_rand, y1_rand, y2_rand, c_dim\
                            = self.DMD_reconstruct(search_resize, current_anno, init_wh, 'patch', dmd_all = False)

                            if  type(current_dmd) == bool:
                                pass
                                
                            else:              
                               
                                y_delta, x_delta = int((y2_rand - y1_rand)*self.cfg.DATA.DMD_AREA_RATE), int((x2_rand - x1_rand)*self.cfg.DATA.DMD_AREA_RATE)

                                plt_dmd = current_dmd[y1_rand+y_delta:y2_rand-y_delta, x1_rand+x_delta:x2_rand-x_delta,c_dim].reshape(-1)
                                plt_draw = int_img[y1_rand+y_delta:y2_rand-y_delta, x1_rand+x_delta:x2_rand-x_delta,c_dim].reshape(-1)

                                # compute mape
                                plt_dmd = plt_dmd.astype(np.int)
                                plt_draw = plt_draw.astype(np.int)
                                mape = np.mean(np.abs(plt_dmd-plt_draw+1e-6) / (np.abs(plt_dmd) + np.abs(plt_draw)+1e-6))
                                if mape < self.cfg.DATA.DMD_MAPE_THRE:
                                    current_frame = cv2.resize(current_dmd,(W,H))
                                    # cv2.rectangle(current_dmd, (int(x1_rand),int(y1_rand)), (int(x2_rand),int(y2_rand)), (0, 255, 0), 2)       
                                    # cv2.imwrite('./draw/dmd_img%02d.jpg'%(int(self.id)), current_dmd) 
                                    # self.id += 1
                                else:
                                    pass
                        else:
                            init_search = data[s + '_images']
                            H, W, C = data[s + '_images'][0].shape
                            data['hw'] = [torch.Tensor((H,W))]
                            
                            search_1 = np.concatenate(init_search, axis = -1)
                            search_resize = cv2.resize(search_1,(self.cfg.DATA.SEARCH_CROP,self.cfg.DATA.SEARCH_CROP))
                                
                            search_resize = search_resize.reshape(self.cfg.DATA.SEARCH_CROP,self.cfg.DATA.SEARCH_CROP, 3, self.cfg.DATA.SEARCH.NUMBER)

                            resize_search = torch.from_numpy(search_resize).float()     
                            data['crop_image'] = resize_search[:,:,:,-self.cfg.DATA.SEARCH.NUMBER:]
                            data['use_multi_frame'] = torch.ones(1)

                    else:
                        data['use_multi_frame'] = torch.zeros(1)
                                           
                    data[s + '_images'] = [current_frame]
                    #cv2.rectangle(current_dmd, (int(x1_rand),int(y1_rand)), (int(x2_rand),int(y2_rand)), (0, 255, 0), 2)       
                    # cv2.imwrite('./draw/dmd_img%02d.jpg'%(int(self.id)), current_frame) 
                    # self.id += 1

                    data[s + '_anno'] = [current_anno]
                    jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
 
                    data['jitter_anno'] = jittered_anno

                    w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                    crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])

                    if (crop_sz < 1).any():
                        
                        data['valid'] = False
                        # print("Too small box is found. Replace it with new data.")
                        return data
                    # Crop image region centered at jittered_anno box and get the attention mask
                    # crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                    #                                                                 data[s + '_anno'], self.search_area_factor[s],
                    #                                                                 self.output_sz[s], masks=data[s + '_masks'])
                    # print('device', data[s + '_images'][0].device)
                    # t3 = time.time()
                    crops, boxes, att_mask, mask_crops, resize_factors = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                    data[s + '_anno'], self.search_area_factor[s],
                                                                                  self.output_sz[s], masks=data[s + '_masks'])
                    # t4 = time.time()
                    # print('crop_time', t4 - t3)

                    boxes_pre = []

                    # print('sear')
                    data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                        image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                    

                # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
                # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
                for ele in data[s + '_att']:
                    
                    if (ele == 1).all():
                        data['valid'] = False
                        # print("Values of original attention mask are all one. Replace it with new data.")
                        return data
                # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
                # for ele in data[s + '_att']:
                    
                #     feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                #     # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                #     mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                #     if (mask_down == 1).all():
                #         data['valid'] = False
                #         # print("Values of down-sampled attention mask are all one. "
                #         #       "Replace it with new data.")
                #         return data
            
            data['valid'] = True
            # if we use copy-and-paste augmentation
            if data["template_masks"] is None or data["search_masks"] is None:
                data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
                data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
            # Prepare output
                
            if self.mode == 'sequence':
                # print('pro_before', len(data['search_images']))
                data = data.apply(stack_tensors)
                # print('pro', len(data['search_images']))
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data
        
        else:
            if self.transform['joint'] is not None:
                data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                    image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
                data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                    image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

            for s in ['template', 'search']:
                assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                    "In pair mode, num train/test frames must be 1"
                
                # if s == 'search':
                #     print('b',data['search_anno'])

                # Add a uniform noise to the center pos
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

                # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
                if (crop_sz < 1).any():
                    data['valid'] = False
                    # print("Too small box is found. Replace it with new data.")
                    return data

                # Crop image region centered at jittered_anno box and get the attention mask
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                                data[s + '_anno'], self.search_area_factor[s],
                                                                                self.output_sz[s], masks=data[s + '_masks'])
                
                # Apply transforms
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

                # if s == 'search':
                #     print('a',data['search_anno'])

                # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
                # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
                for ele in data[s + '_att']:
                    if (ele == 1).all():
                        data['valid'] = False
                        # print("Values of original attention mask are all one. Replace it with new data.")
                        return data
                # # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
                for ele in data[s + '_att']:
                    feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                    # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                    mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                    if (mask_down == 1).all():
                        data['valid'] = False
                        # print("Values of down-sampled attention mask are all one. "
                        #       "Replace it with new data.")
                        return data

            data['valid'] = True
            # if we use copy-and-paste augmentation
            if data["template_masks"] is None or data["search_masks"] is None:
                data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
                data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
            # Prepare output
            if self.mode == 'sequence':
                data = data.apply(stack_tensors)
            else:
                data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

            return data
        

