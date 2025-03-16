import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize
import time
'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''
def sample_crop(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :, :]
    pad_list = [y1_pad, y2_pad, x1_pad, x2_pad]
    return im_crop, pad_list

def sample_pad(im, pad_list, output_sz):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    y1_pad, y2_pad, x1_pad, x2_pad = pad_list[0], pad_list[1], pad_list[2], pad_list[3]
    # Pad
    im_crop_padded = cv.copyMakeBorder(im, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
    return im_crop_padded
    # deal with attention mask

def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded
    
def sample_target_4corner(im, target_bb, search_area_factor, output_sz=None, mask=None):
   
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    jitter_list = [(0, 0), (crop_sz/3, crop_sz/3), (crop_sz/3, -crop_sz/3),\
                   (-crop_sz/3, crop_sz/3), (-crop_sz/3, -crop_sz/3)]

    if crop_sz < 1:
        raise Exception('Too small bounding box.')
    
    img_crop_all = []
    for jitter in jitter_list:
        x1 = round(x + 0.5 * w - crop_sz * 0.5 + jitter[0])
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5 + jitter[1])
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
        
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        img_crop_all.append(im_crop_padded)
    img_crop_all = np.stack(img_crop_all, axis = 0)
    return img_crop_all, resize_factor, jitter_list
        
   
def sample_target_numpy(im, target_bb, search_area_factor, output_size, hw, device):

    B = im.shape[0]
    
    h_c, w_c = im.shape[-2], im.shape[-1]
    im = im.permute(0, 2, 3, 1)

    target_bb = target_bb.squeeze(0)
    hw = hw.squeeze(0)

    im, target_bb, hw = \
    im.cpu().numpy(),  target_bb.cpu().numpy(), hw.cpu().numpy()

    h_b, w_b = hw[:, 0], hw[:, 1]
    f_h, f_w = h_c / h_b, w_c / w_b
    
    x, y, w, h = target_bb[:,0]*f_w, target_bb[:,1]*f_h, target_bb[:,2]*f_w, target_bb[:,3]*f_h

    out = []
    for i in range(B):
    
        # Crop image
        crop_sz = math.ceil(math.sqrt(w[i] * h[i]) * search_area_factor)
        # crop_sz = (torch.sqrt(w * h) * search_area_factor).ceil() #up int

        # if crop_sz < 1:
        #     raise Exception('Too small bounding box.')

        x1 = round(x[i] + 0.5 * w[i] - crop_sz * 0.5)
        # x1 = x + 0.5 * w - crop_sz * 0.5
        x2 = x1 + crop_sz

        y1 = round(y[i] + 0.5 * h[i] - crop_sz * 0.5)
        # y1 = y + 0.5 * h - crop_sz * 0.5
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - w_c + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - h_c + 1, 0)

        imx = im[i,:,:,:]
    # Crop target
        im_crop = imx[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
        im_crop_padded = cv.resize(im_crop_padded, (output_size, output_size))
        out.append(im_crop_padded)

    out = np.stack(out, axis = 0) # b, h, w, c
    out = torch.Tensor(out).permute(0, 3, 1, 2).to(device)
        
    return out

def sample_wave_index(im, target_bb, search_area_factor, output_resize, hw, zeros_input = None,
                      patch_size = 16, stride = 16, thre = 0.8, id = None):
    
    zeros = zeros_input
    # print('im', im.shape)
    h_c, w_c = im.shape[0], im.shape[1]
    hw, target_bb = hw.squeeze(0), target_bb.squeeze(0)
    # print(target_bb.shape, hw.shape)
    h_b, w_b = hw[0], hw[1]
    f_h, f_w = h_c / h_b, w_c / w_b
    
    x, y, w, h = target_bb[0]*f_w, target_bb[1]*f_h, target_bb[2]*f_w, target_bb[3]*f_h

    crop_sz = (torch.sqrt(w * h) * search_area_factor).ceil() #up int

    x1 = torch.round(x + 0.5 * w - crop_sz * 0.5).float()
    x2 = x1 + crop_sz

    y1 = torch.round(y + 0.5 * h - crop_sz * 0.5).float()
    y2 = y1 + crop_sz

    x1_pad = torch.max(zeros, -x1)
    x2_pad = torch.max(x2 - w_c + 1, zeros)

    y1_pad = torch.max(zeros, -y1)
    y2_pad = torch.max(y2 - h_c + 1, zeros)
 
    x1_p = int(x1_pad.item())
    x2_p = int(x2_pad.item())
    y1_p = int(y1_pad.item())
    y2_p = int(y2_pad.item())

    x1_t = int(x1.item())
    x2_t = int(x2.item())
    y1_t = int(y1.item())
    y2_t = int(y2.item())

    img_i = im[y1_t + y1_p:y2_t - y2_p, x1_t + x1_p : x2_t - x2_p, :].permute(2, 0, 1)

    # print('img_i', img_i.shape)
    # print(x1_p, x2_p, y1_p, y2_p)
    img_pad = F.pad(img_i,(x1_p, x2_p, y1_p, y2_p),'constant',0) # h, w, c
    # print(img_pad.shape)

    output_img = output_resize(img_pad)
    # print('output_img', output_img.shape)
    # save_path = './save/'+id+'_wave_crop.jpg'  
    # image1 = output_img.permute(1, 2, 0).clone()
    # max_value = torch.max(image1, dim = 0)[0]
    # image1 = image1 / max_value * 255
    # image1 = torch.sum(image1, dim = -1)
    # image1 = image1.cpu().numpy()
    # cv.imwrite(save_path, image1)
    # print(output_img.shape)

    unfold_h = output_img.unfold(1, patch_size, stride)
    unfold_w = unfold_h.unfold(2, patch_size, stride)
    unfold_w = torch.abs(unfold_w)
    # print('unfold_w', unfold_w.shape)
    unfold_w = torch.sum(unfold_w, dim = [0, -1, -2])

    result = unfold_w.flatten()
    # print('result', result)
    index_all = torch.argsort(result, dim = 0, descending = True)
    # print('index_all', index_all)
    len_index = int(thre * index_all.shape[0])
    index_wave = index_all[:len_index]
    rest_wave = index_all[len_index:]
    
    return index_wave, rest_wave
       
def sample_target_tensor(im, target_bb, search_area_factor, output_resize, hw, zeros_input = None):
    # print(im.shape) # 32, 3, 512, 512
    # print(target_bb.shape)
    
    # print(all_anno.shape) # 1, 32, 4
    # print(jitter.shape) # 1, 32, 4
    # hw torch.Size([1, 32, 2])
    
    B = im.shape[0]
    # zeros =torch.zeros(B).to(im.device)
    zeros = zeros_input
    
    h_c, w_c = im.shape[-2], im.shape[-1]
    if len(target_bb.shape) == 3:
        target_bb = target_bb.squeeze(0)
        hw = hw.squeeze(0)

    h_b, w_b = hw[:, 0], hw[:, 1]
    f_h, f_w = h_c / h_b, w_c / w_b
    
    x, y, w, h = target_bb[:,0]*f_w, target_bb[:,1]*f_h, target_bb[:,2]*f_w, target_bb[:,3]*f_h

    crop_sz = (torch.sqrt(w * h) * search_area_factor).ceil() #up int

    x1 = torch.round(x + 0.5 * w - crop_sz * 0.5).float()
    # x1 = x + 0.5 * w - crop_sz * 0.5
    x2 = x1 + crop_sz

    y1 = torch.round(y + 0.5 * h - crop_sz * 0.5).float()
    # y1 = y + 0.5 * h - crop_sz * 0.5
    y2 = y1 + crop_sz

    x1_pad = torch.max(zeros, -x1)
    # x1_pad = torch.where(-x1 >0, -x1, 0)
    # x1_pad = -x1*(torch.gt(-x1, 0).int())
    x2_pad = torch.max(x2 - w_c + 1, zeros)
    # x2_pad = torch.where(x2>(w_c-1), x2 - w_c + 1, 0)
    # x2_pad = 

    y1_pad = torch.max(zeros, -y1)
    y2_pad = torch.max(y2 - h_c + 1, zeros)

    # t1 = time.time()
    for i in range(B):
         #b, c, h, w
        x1_p = int(x1_pad[i].item())
        x2_p = int(x2_pad[i].item())
        y1_p = int(y1_pad[i].item())
        y2_p = int(y2_pad[i].item())

        x1_t = int(x1[i].item())
        x2_t = int(x2[i].item())
        y1_t = int(y1[i].item())
        y2_t = int(y2[i].item())

        img_i = im[i, :, y1_t + y1_p:y2_t - y2_p, x1_t + x1_p : x2_t - x2_p]

        # p1 = time.time()
        img_pad = F.pad(img_i,(x1_p, x2_p, y1_p, y2_p),'constant',0).unsqueeze(0)
        # p2 = time.time()
        # print('pad', p2 -p1)

        # tp1 = time.time()
        output_img = output_resize(img_pad)
        # print('output_img', output_img.shape)
        # tp2 = time.time()
        # print('resize_32', tp2 -tp1)

        if i == 0:
            out = output_img
        else:
            out = torch.cat([out, output_img], dim = 0)
    
    # t2 = time.time()
    # print('enu', t2 -t1)

    return out

def sample_target_single(im, target_bb, search_area_factor, output_resize, hw, zeros):
    # print(img.shape) # 1, 3, 512, 512
    
    # print(jitter.shape) # , 1, 4
    # hw torch.Size([ 1, 2])
    
    B = im.shape[0]

    # print(target_bb.shape, hw.shape)
    h_c, w_c = im.shape[-2], im.shape[-1]
    target_bb = target_bb
    hw = hw
    h_b, w_b = hw[0], hw[1]
    f_h, f_w = h_c / h_b, w_c / w_b
    
    x, y, w, h = target_bb[0]*f_w, target_bb[1]*f_h, target_bb[2]*f_w, target_bb[3]*f_h

    crop_sz = (torch.sqrt(w * h) * search_area_factor).ceil() #up int

    x1 = torch.round(x + 0.5 * w - crop_sz * 0.5).float()
    # x1 = x + 0.5 * w - crop_sz * 0.5
    x2 = x1 + crop_sz

    y1 = torch.round(y + 0.5 * h - crop_sz * 0.5).float()
    # y1 = y + 0.5 * h - crop_sz * 0.5
    y2 = y1 + crop_sz

    x1_pad = torch.max(zeros, -x1)
    # x1_pad = torch.where(-x1 >0, -x1, 0)
    # x1_pad = -x1*(torch.gt(-x1, 0).int())
    x2_pad = torch.max(x2 - w_c + 1, zeros)
    # x2_pad = torch.where(x2>(w_c-1), x2 - w_c + 1, 0)
    # x2_pad = 

    y1_pad = torch.max(zeros, -y1)
    y2_pad = torch.max(y2 - h_c + 1, zeros)

    t1 = time.time()
    for i in range(B):
         #b, c, h, w
        x1_p = int(x1_pad.item())
        x2_p = int(x2_pad.item())
        y1_p = int(y1_pad.item())
        y2_p = int(y2_pad.item())

        x1_t = int(x1.item())
        x2_t = int(x2.item())
        y1_t = int(y1.item())
        y2_t = int(y2.item())

        img_i = im[i, :, y1_t + y1_p:y2_t - y2_p, x1_t + x1_p : x2_t - x2_p]

        # p1 = time.time()
        img_pad = F.pad(img_i,(x1_p, x2_p, y1_p, y2_p),'constant',0).unsqueeze(0)
        # p2 = time.time()
        # print('pad', p2 -p1)

        # tp1 = time.time()
        output_img = output_resize(img_pad)
        # tp2 = time.time()
        # print('resize_32', tp2 -tp1)

        if i == 0:
            out = output_img
        else:
            out = torch.cat([out, output_img], dim = 0)
    
    # t2 = time.time()
    # print('enu', t2 -t1)

    return out

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, center:torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False,use_predict = False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    # print('box_extract_center',box_extract_center)

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    # print('box_in_center',box_in_center)
    if not use_predict:
        box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
        # print('crop_sz',crop_sz)
        # print('resize_factor',resize_factor)
    else:
        box_out_center = (crop_sz - 1) / 2 + (box_in_center - center) * resize_factor

    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    # print('box_out',box_out)
    if normalize:
        # print('box_out_nor',box_out/ crop_sz[0])
        return box_out / crop_sz[0]
    else:
        return box_out

def transform_image_to_crop_true(box_in: torch.Tensor, box_extract: torch.Tensor, center:torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor,box_true:torch.Tensor, normalize=False,use_predict = False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    box_in_true_center = box_true[0:2] + 0.5 * box_true[2:4]
    if not use_predict:
        box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
        box_out_true_center = (crop_sz - 1) / 2 + (box_in_true_center - box_extract_center) * resize_factor
    else:
        box_out_center = (crop_sz - 1) / 2 + (box_in_center - center) * resize_factor

    box_out_wh = box_in[2:4] * resize_factor
    box_out_true_wh = box_true[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    box_out_true = torch.cat((box_out_true_center - 0.5 * box_out_true_wh, box_out_true_wh))
    # print('box_out',box_out)
    if normalize:
        # print('box_out_nor',box_out/ crop_sz[0])
        return box_out / crop_sz[0],box_out_true / crop_sz[0]
    else:
        return box_out,box_out_true


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    # data[s + '_images'], jittered_anno,
    #                             frame, self.search_area_factor[s],
    #                                 self.output_sz[s], masks=data[s + '_masks']
    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])
    # print('crop_sz',crop_sz)
    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    #
    box_crop = [transform_image_to_crop(a_gt, a_ex, None, rf, crop_sz, normalize=True,use_predict=False)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]
    
    # box_no_normal = [transform_image_to_crop(a_gt, a_ex, None, rf, crop_sz, normalize=False,use_predict=False)
    #             for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]
    
      # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop,resize_factors

def jittered_center_crop_true(frames, box_extract, box_gt, search_area_factor, output_sz, box_true, masks=None):
    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])
    # print('crop_sz',crop_sz)
    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    # box_crop = [transform_image_to_crop(a_gt, a_ex, None, rf, crop_sz, normalize=True,use_predict=False)
    #             for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors
    box_all = [transform_image_to_crop_true(a_gt, a_ex, None, rf, crop_sz, a_true,normalize=True,use_predict=False)
                for a_gt, a_ex, rf,a_true in zip(box_gt, box_extract, resize_factors,box_true)]
    # print(box_all) 
    # exit(-1)
    # box_crop,box_crop_true
    return frames_crop, [box_all[0][0]],[box_all[0][1]], att_mask, masks_crop


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

