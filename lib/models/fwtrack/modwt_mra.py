import pywt 
import numpy as np
import cv2
import os
import torch
import time
import torch.nn.functional as F

from scipy.ndimage import convolve1d


def circular_convolve_d(h_t, v_j_1, j):
    """
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    """
    N = len(v_j_1)
    w_j = np.zeros(N)
    ker = np.zeros(len(h_t) * 2**(j - 1))

    # make kernel
    for i, h in enumerate(h_t):
        ker[i * 2**(j - 1)] = h

    w_j = convolve1d(v_j_1, ker, mode="reflect", origin=-len(ker) // 2)
    return w_j

def modwt(x, filters, level):
    """
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    """
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return wavecoeff

def get_kernel(g, level = 1):
    ker = np.zeros(len(g) * 2**(level - 1))

    # make kernel
    for i, h in enumerate(g):
        ker[i * 2**(level - 1)] = h

    return ker

def get_ker(use_time = False):

    w = pywt.Wavelet('db5')
    h = np.array(w.dec_hi) / np.sqrt(2) 
    g = np.array(w.dec_lo) / np.sqrt(2) 

    g_1, g_2, g_3 = get_kernel(g, 1), get_kernel(g, 2), get_kernel(g, 3)

    g_hw = [np.flip(g_1).copy(), np.flip(g_2).copy(), np.flip(g_3).copy()]
     
    return g_hw

def get_ker_level(level = 3, type_k = 'smooth'):

    w = pywt.Wavelet('db5')
    h = np.array(w.dec_hi) / np.sqrt(2) 
    g = np.array(w.dec_lo) / np.sqrt(2) 

    g_hw = []
    for i in range(level):
        if type_k == 'smooth':
            gi = get_kernel(g, i+1)
            g_hw.append(np.flip(gi).copy())
        
        else:
            hi = get_kernel(h, i+1)
            g_hw.append(np.flip(hi).copy())
     
    return g_hw
   
    
def get_ker_diff():

    w = pywt.Wavelet('db5')
    h = np.array(w.dec_hi) / np.sqrt(2) 
    g = np.array(w.dec_lo) / np.sqrt(2) 

    h_1, h_2, h_3 = get_kernel(h, 1), get_kernel(h, 2), get_kernel(h, 3)
    
    h_hw = [np.flip(h_1).copy(), np.flip(h_2).copy(), np.flip(h_3).copy()]
        
    return h_hw

def get_ker_deep(use_time = False):

    w = pywt.Wavelet('haar')
    h = np.array(w.dec_hi) / np.sqrt(2) 
    g = np.array(w.dec_lo) / np.sqrt(2) 

    g_1, g_2, g_3 = get_kernel(g, 1), get_kernel(g, 2), get_kernel(g, 3)

    g_hw = [np.flip(g_1).copy(), np.flip(g_2).copy(), np.flip(g_3).copy()]
    if not use_time:
        
        return g_hw
    else:
        w_haar = pywt.Wavelet('haar')
        h_haar = np.array(w_haar.dec_hi) / np.sqrt(2) 
        g_haar = np.array(w_haar.dec_lo) / np.sqrt(2)    
        g_haar1, g_haar2, g_haar3 = get_kernel(g_haar, 1), get_kernel(g_haar, 2), get_kernel(g_haar, 3)
        g_t = [g_haar1, g_haar2, g_haar3]
        return g_hw, g_t
    

def modwt_2d(x, g, weight, device):

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    H, W, B, C = x.shape[2], x.shape[3], x.shape[0], x.shape[1]
    kernel_size = len(g)
    
    x = x.permute(0, 1, 3, 2).contiguous().view(B*C*W, H) # 3, B, W, H
    pad_left = kernel_size-1
    pad_right = 0

    padding_x_0 = torch.cat([x[:,-pad_left:], x], dim = 1)
    
    x_h = padding_x_0.unsqueeze(1)
    result_H = F.conv1d(x_h, weight, padding = 'valid') # B*3*W, 1, H_pad -> B*3*W, 1, H

    result_h1 = result_H.squeeze().view(B, C, W, H).permute(0, 1, 3, 2).contiguous().view(B*C*H, W)
    
    padding_x_1 = torch.cat([result_h1[:,-pad_left:], result_h1], dim = 1)

    x_w = padding_x_1.unsqueeze(1)
    result_W = F.conv1d(x_w, weight, padding = 'valid').squeeze() #B*3*H, W
    
    x_result = result_W.view(B, C, H, W) #output: B, 3, H, W
    return x_result

def modwt_2d_hw(x, g, g1, weight, weight1, device):
    # input:  (batch, 3, H, W)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    H, W, B, C = x.shape[2], x.shape[3], x.shape[0], x.shape[1]
    kernel_size = len(g)
    
    x = x.permute(0, 1, 3, 2).contiguous().view(B*C*W, H) # 3, B, W, H
    pad_left = kernel_size-1

    padding_x_0 = torch.cat([x[:,-pad_left:], x], dim = 1)
    
    x_h = padding_x_0.unsqueeze(1)
    result_H = F.conv1d(x_h, weight, padding = 'valid') # B*3*W, 1, H_pad -> B*3*W, 1, H
    
    result_h1 = result_H.squeeze().view(B, C, W, H).permute(0, 1, 3, 2).contiguous().view(B*C*H, W)
   
    padding_x_1 = torch.cat([result_h1[:,-pad_left:], result_h1], dim = 1)

    x_w = padding_x_1.unsqueeze(1)
    result_W = F.conv1d(x_w, weight1, padding = 'valid').squeeze() #B*3*H, W
    
    x_result = result_W.view(B, C, H, W) #output: B, 3, H, W
    return x_result

def get_modwt2d(test_img, g_hw, weight_hw, device):
 
    g_1, g_2, g_3 = g_hw[0], g_hw[1], g_hw[2]
    w_1, w_2, w_3 = weight_hw[0], weight_hw[1], weight_hw[2]
    x1 = modwt_2d(test_img, g_1, w_1, device)

    x2 = modwt_2d(x1, g_2, w_2, device)

    x3 = modwt_2d(x2, g_3, w_3, device)
    
    x_result_all = torch.cat([x1, x2, x3], dim = 1)
    return x_result_all

def get_modwt2d_level(test_img, g_hw, weight_hw, device, level = 3):
 
    x_result_all = []
    xi = test_img
    for i in range(level):

        xi = modwt_2d(xi, g_hw[i], weight_hw[i], device)
        x_result_all.append(xi)

    return x_result_all

def modwt2d_v_w_level(test_img, g, h, weight_g, weight_h, device, level = 2, only_w3 = False):

    x_result_all = []
    v = test_img
    for i in range(level):
        if only_w3:
            w3 = modwt_2d_hw(v, h[i], h[i], weight_h[i], weight_h[i], device)
            v = modwt_2d_hw(v, g[i], g[i], weight_g[i], weight_g[i], device)
            x_result_all.append(w3)
        
        else:
            w1 = modwt_2d_hw(v, g[i], h[i], weight_g[i], weight_h[i], device)
            w2 = modwt_2d_hw(v, h[i], g[i], weight_h[i], weight_g[i], device)
            w3 = modwt_2d_hw(v, h[i], h[i], weight_h[i], weight_h[i], device)
            v = modwt_2d_hw(v, g[i], g[i], weight_g[i], weight_g[i], device)

            x_result_all.append(v)
            x_result_all.append(w1)
            x_result_all.append(w2)
            x_result_all.append(w3)

    return x_result_all

def get_modwt2d_for_difference(test_img, h_hw, weight_hw, device, level = 3):
    
    w_all = []
    for i in range(level):
        wi = modwt_2d(test_img[i], h_hw[i], weight_hw[i], device)
        w_all.append(wi)
    return w_all

def get_modwt2d_single(test_img, g_hw, weight_hw, device):
 
    g_1 = g_hw[0]
    w_1 = weight_hw[0]
    
    x1 = modwt_2d(test_img, g_1, w_1, device)
    
    return x1

def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li
    
def reconstruct(x, g_weight, device):
   
    B, C, H, W = x.shape
  
    pad_right = (g_weight.shape[-1])-1
    x = x.contiguous().view(B*C*H, W)
    
    padding_x_0 = torch.cat([x, x[:,0:pad_right]], dim = 1)
   
    weight1 = g_weight
    
    x_0 = padding_x_0.unsqueeze(1)
    # print('x_0', x_0.shape)
    result = F.conv1d(x_0, weight1, padding = 'valid')
    # print('result', result.shape)

    result_w = result.squeeze().view(B, C, H, W).permute(0, 1, 3, 2).contiguous().view(B*C*W, H)

    
    padding_x_1 = torch.cat([result_w, result_w[:,0:pad_right]], dim = 1)

    x_w = padding_x_1.unsqueeze(1)
    result_W = F.conv1d(x_w, weight1, padding = 'valid').squeeze() #B*3*H, W
    
    x_result = result_W.contiguous().view(B, C, W, H).permute(0, 1, 3, 2) #output: B, 3, H, W
    t2 = time.time()
    # print('construct_conv', t2 - t1)

    return x_result

def get_modwt_mra(test_img, g_hw, device):
    x1, x2, x3 = test_img[:,0:3,:,:], test_img[:,3:6,:,:], test_img[:,6:9,:,:]
    g_1, g_2, g_3 = g_hw[0], g_hw[1], g_hw[2]
    # level 1
    result1 = reconstruct(x1, g_1, device)
    # level 2
    result2 = reconstruct(x2, g_2, device)
    # level 3
    result3 = reconstruct(x3, g_3, device)

    result_all = torch.cat([result1, result2, result3], dim = 1)
    return result_all

def get_modwt_mra_level(test_img, g_hw, device, level = 5):
    result_all = []
    for i in range(level):
  
        result_i = reconstruct(test_img[i], g_hw[i], device)
        result_all.append(result_i)
  
    return result_all

def get_modwt_mra_single(test_img, g_hw, device):
    
    g_1 = g_hw[0]
    result1 = reconstruct(test_img, g_1, device)
    
    return result1

def get_smooth_kernel(g, level):
    g_j_part = [1]
    for j in range(level):   
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)

    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)  
    g_j_t = g_j / (2**((j + 1) / 2.))
    
    return g_j_t

def get_all_smooth_kernel(level = 3):
    w = pywt.Wavelet('db5')

    g = np.array(w.dec_lo)
    gx = []
    for i in range(level):
        gi = get_smooth_kernel(g, i+1)
        gx.append(gi)

    return gx

def get_mra_kernel_weight(gx, img_size, device, level):
    weight_list = []
    for i in range(level):
        gi = period_list(gx[i], img_size)
        with torch.no_grad(): 
            weighti = torch.Tensor(gi).unsqueeze(0).unsqueeze(0).to(device)
        weight_list.append(weighti)
    return weight_list

def get_all_smooth_kernel_deep():
    w = pywt.Wavelet('haar')

    g = np.array(w.dec_lo)
    g1 = get_smooth_kernel(g, 1)
    g2 = get_smooth_kernel(g, 2)
    g3 = get_smooth_kernel(g, 3)

    return [g1, g2, g3]

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2**(j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2**(j - 1) * i] = li[i]
    return li_n
