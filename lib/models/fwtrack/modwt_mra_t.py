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
    filters: 'db1', 'db2', 'db2', ...
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


def get_ker_level_t(level = 2, type_k = 'smooth'):

    w = pywt.Wavelet('db2')
    # w = pywt.Wavelet('haar')
    h = np.array(w.dec_hi) / np.sqrt(2) # 高频
    g = np.array(w.dec_lo) / np.sqrt(2) # 低频

    g_hw = []
    for i in range(level):
        if type_k == 'smooth':
            gi = get_kernel(g, i+1)
            g_hw.append(np.flip(gi).copy())
        
        else:
            hi = get_kernel(h, i+1)
            g_hw.append(np.flip(hi).copy())
     
    return g_hw
 
def get_ker_deep(use_time = False):

    w = pywt.Wavelet('db2')
    h = np.array(w.dec_hi) / np.sqrt(2) # 高频
    g = np.array(w.dec_lo) / np.sqrt(2) # 低频

    g_1, g_2, g_3 = get_kernel(g, 1), get_kernel(g, 2), get_kernel(g, 3)
    # print(g_1)
    # print(np.flip(g_1))
    # reverse
    g_hw = [np.flip(g_1).copy(), np.flip(g_2).copy(), np.flip(g_3).copy()]
    if not use_time:
        
        return g_hw

def modwt_2d(x, g, weight, device):

    kernel_size = len(g)

    pad_left = kernel_size-1
    pad_right = 0

    padding_x_0 = torch.cat([x[:,-pad_left:], x], dim = 1)
  
    x_h = padding_x_0.unsqueeze(1)
    result_H = F.conv1d(x_h, weight, padding = 'valid') # B*3*W, 1, H_pad -> B*3*W, 1, H
   
    result_h1 = result_H.squeeze()

    return result_h1

def get_modwt2d_t(test_img, g_hw, weight_hw, device):
 
    g_1, g_2, g_3 = g_hw[0], g_hw[1], g_hw[2]
    w_1, w_2, w_3 = weight_hw[0], weight_hw[1], weight_hw[2]
    
    x1 = modwt_2d(test_img, g_1, w_1, device)
 
    x2 = modwt_2d(x1, g_2, w_2, device)
   
    return x1, x2

def modwt2d_t_level(test_img, g, h, weight_g, weight_h, device, level = 2):
 
    img_list = []
    v = test_img
    for i in range(level):
        w = modwt_2d(v, h[i], weight_h[i], device)
        v = modwt_2d(v, g[i], weight_g[i], device)
        img_list.append(w)
    output = torch.stack(img_list, dim = 0)
    return output
    

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
    
def reconstruct(x, g, device):

    N, t = x.shape
    g_h = period_list(g, t)
    
    pad_left = 0
    pad_right = (len(g_h))-1

    padding_x_0 = torch.cat([x, x[:,0:pad_right]], dim = 1)
    with torch.no_grad(): 
        weight1 = torch.Tensor(g_h).unsqueeze(0).unsqueeze(0).to(device)

    x_0 = padding_x_0.unsqueeze(1)
    result = F.conv1d(x_0, weight1, padding = 'valid')

    result_w = result.squeeze()

    return result_w

def get_modwt_mra_t(test_img, g_hw, device):
    x3 = test_img

    g_1, g_2, g_3 = g_hw[0], g_hw[1], g_hw[2]

    result3 = reconstruct(x3, g_3, device)

    return result3

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

def get_all_smooth_kernel_t():
    w = pywt.Wavelet('db2')

    g = np.array(w.dec_hi)
    g1 = get_smooth_kernel(g, 1)
    g2 = get_smooth_kernel(g, 2)
    g3 = get_smooth_kernel(g, 3)

    return [g1, g2, g3]

def get_all_smooth_kernel_deep():
    w = pywt.Wavelet('db2')

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

