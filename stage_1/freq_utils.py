import os
import os.path
import argparse
import numpy as np
import cv2
import glob
import random
import matplotlib.pyplot as plt
import torch

nx, ny = (256, 256)
x = np.linspace(-1,1,nx)
y = np.linspace(-1,1, ny)
xv, yv = np.meshgrid(x, y)



def plot_all_funcs(x_im, dx=32, half=False):
    # z = ((z+1)/2 * mask - 0.5)/0.5
    color=False
    if len(x_im.shape)>=3:
        color=True
    
    mask = np.ones_like(x_im)
    w, h = mask.shape
    x0, y0 = 0, 0
    if half:
        mask[:, 0:w//2]=0
    else: # for center
        mask[h//2- x0 - dx: h//2 - x0 + dx, w//2 - y0 - dx: w//2 - y0 + dx]=0
#     z_mask = z[:,:,0] * mask#
    if color:
        x_im = x_im[:,:,0]
    else:
        pass
        
    im_mask = (((x_im + 1)/2 * mask) - 0.5) / 0.5
        
    dct1_cv, mag_cv, phase_cv, idx_dct_cv = fft_compute(x_im, center=True) 
    min_v1 = mag_cv.min()
    max_v1 = mag_cv.max()
    
    dct1_cv_m, mag_cv_m, phase_cv_m, idx_dct_cv_m = fft_compute(im_mask, center=True) 
    min_v2 = mag_cv.min()
    max_v2 = mag_cv.max()
    
    min_v = min(min_v1, min_v2)
    max_v = max(max_v1, max_v2)
    
    mag_cv = (mag_cv - min_v)/(max_v - min_v)
    phase_cv = phase_cv/(2 * np.pi)
    
    mag_cv_m = (mag_cv_m - min_v)/(max_v - min_v)
    phase_cv_m = phase_cv_m/(2 * np.pi)
    
    plt.imshow(np.concatenate((mag_cv, mag_cv_m), axis=1), cmap='ocean')
    plt.colorbar()
    plt.title('Power spectrum')
    plt.show()

    plt.imshow(np.concatenate((phase_cv, phase_cv_m), axis=1), cmap='ocean')
    plt.colorbar()
    plt.title('Phase spectrum')
    plt.show()


    img_back = ifft_compute(mag_cv, phase_cv, idx_dct_cv, center=True, val_lims=[min_v, max_v])
    img_back_masked = ifft_compute(mag_cv_m, phase_cv_m, idx_dct_cv_m, center=True, val_lims=[min_v, max_v])

    plt.imshow(np.concatenate(((x_im+1)/2, (im_mask+1)/2), axis=1), cmap='gray')
    plt.colorbar()
    plt.title('Original image')
    plt.show()

    plt.imshow(np.concatenate(((img_back+1)/2, (img_back_masked+1)/2), axis=1), cmap='gray')
    plt.colorbar()
    plt.title('Recon image')
    plt.show()



#######################################################################
###############         grayscale utils here            ###############
#######################################################################

def fft_compute(img, center=False):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    if center:
        dft = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft[:,:,0],dft[:,:,1])
    idx = (mag==0)
    mag[idx] = 1.
    magnitude_spectrum = np.log(mag)
    phase_spectrum = cv2.phase(dft[:,:,0],dft[:,:,1])
    return dft, magnitude_spectrum, phase_spectrum, idx

def ifft_compute(magnitude, phase, idx, center=False, val_lims=None):
    min_v, max_v = val_lims
    
    recon_mag = magnitude.copy()
    recon_mag = (max_v - min_v) * recon_mag + min_v
    recon_mag[idx] = -np.inf
    
    recon_phase = phase * (2 * np.pi)
    
    real_part = np.cos(recon_phase) * np.exp(recon_mag)
    imag_part = np.sin(recon_phase) * np.exp(recon_mag)
   
    true_fft = np.concatenate((real_part[:, :, None], imag_part[:, :, None]), axis=-1)
    if center:
        true_fft = np.fft.fftshift(true_fft)
    img_back = cv2.idft(true_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return img_back

    
# fx=10
# fy=12
# # z = (np.cos(2*np.pi*(fx*xv + fy*yv)) + 1)/2
# z = np.cos(2*np.pi*(fx*xv + fy*yv))
def get_gray_fft_images(x_im, dx=16, half=False, return_mask=False):
    x_im = np.squeeze(x_im, axis=1)
    assert len(x_im[0].shape)<=2, "Image should be grayscale"
    h, w = x_im[0].shape
    mask = np.ones((h,w))
    
    if half:
        mask[:, 0:w//2]=0
    else:
        mask[h//2-dx: h//2+dx, w//2-dx: w//2+dx]=0
    
    N = len(x_im)
    x_masked = np.zeros((N, 1, h, w))
    x_fft = np.zeros((N, 2, h, w))
    x_masked_fft = np.zeros((N, 2, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i,0] = x_m
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,0] = mag_x_
        x_fft[i,1] = phase_x_
        
        x_masked_fft[i,0] = mag_x_m
        x_masked_fft[i,1] = phase_x_m
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)
    
    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, mask
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m

def get_gray_images_back(x_fft, lims_list, idx_list):
    h, w = x_fft[0, 0].shape
    
    N = len(x_fft)
    x_back = np.zeros((N, 1, h, w))

    for i in range(N):
        mag_ = x_fft[i, 0]
        phase_ = x_fft[i, 1]
        idx_ = idx_list[i]
        lims_ = lims_list[i]
        
        img_back = ifft_compute(mag_, phase_, idx_, center=True, val_lims=lims_)
        x_back[i,0]=img_back
        
    return x_back
        
    
#######################################################################
###############             Color utils here            ###############
#######################################################################

def fft_compute_color(img_col, center=False):
    assert img_col.shape[0]==3, "Should be color image"
    _, h, w = img_col.shape
    lims_list = []
    idx_list_ = []
    x_mag = np.zeros((3, h, w))
    x_phase = np.zeros((3, h, w))
    x_fft = np.zeros((6, h, w))
    
    for i in range(3):
        img = img_col[i]
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        if center:
            dft = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft[:,:,0],dft[:,:,1])
        idx = (mag==0)
        mag[idx] = 1.
        magnitude_spectrum = np.log(mag)
        phase_spectrum = cv2.phase(dft[:,:,0],dft[:,:,1])
        x_mag[i] = magnitude_spectrum
        x_phase[i] = phase_spectrum
        
        x_fft[2*i] = dft[:,:,0]
        x_fft[2*i+1] = dft[:,:,1]
        
        idx_list_.append(idx)
    
    return x_fft, x_mag, x_phase, idx_list_


    
    
def get_color_fft_images(x_im, dx=16, half=False, return_mask=False): # b x 3 x h x w 
#     x_im = np.squeeze(x_im, axis=1)
    # x_fft has first three channel as magnitude next 3 channels as phase
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w
    h, w = x_im[0].shape[1:]
    mask = np.ones((3,h,w))
    
    if half:
        mask[:, :, 0:w//2]=0
    else:
        mask[:, h//2-dx: h//2+dx, w//2-dx: w//2+dx]=0
    
    N = len(x_im)
    x_masked = np.zeros((N, 3, h, w))
    x_fft = np.zeros((N, 6, h, w))
    x_masked_fft = np.zeros((N, 6, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i] = x_m
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute_color(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute_color(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,:3] = mag_x_
        x_fft[i,3:] = phase_x_
        
        x_masked_fft[i, :3] = mag_x_m
        x_masked_fft[i, 3:] = phase_x_m
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)

    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, mask
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m

    
########  new ######
def make_one_masked_image_new(x_im, mask, return_mask=True): # 1 x 3 x h x w 
    # x_fft has first three channel as magnitude next 3 channels as phase
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w
    h, w = x_im[0].shape[1:]
    
    # all_masks = mask# np.ones((1, h, w, 3))
    # mask = cv2.resize(np.array(Image.open(mask_file)),(w,h))
    all_masks=mask
    
    x_masked = np.zeros((N, 3, h, w))
    x_fft = np.zeros((N, 6, h, w))
    x_masked_fft = np.zeros((N, 6, h, w))
    mask_fft = np.zeros((N, 6, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        mask = np.transpose(all_masks[i], (2,0,1))
        # import ipdb; ipdb.set_trace()
        x_m = (((x_ + 1)/2 * mask) - 0.5) / 0.5
        x_masked[i] = x_m
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute_color(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute_color(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,:3] = mag_x_
        x_fft[i,3:] = phase_x_
        
        x_masked_fft[i, :3] = mag_x_m
        x_masked_fft[i, 3:] = phase_x_m
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)

    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, mask
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m
        
        
#         ##############################################################################
#         _, mag_fft_mask_, phase_fft_mask_x_, _ = fft_compute_color(mask, center=True) 
#         min_v3 = mag_fft_mask_.min()
#         max_v3 = mag_fft_mask_.max()
#         mag_fft_mask_ = (mag_fft_mask_ - min_v3)/(max_v3 - min_v3)
#         phase_fft_mask_x_ = phase_fft_mask_x_/(2 * np.pi)
        
#         mask_fft[i,:3] = mag_fft_mask_
#         mask_fft[i,3:] = phase_fft_mask_x_
#         ###############################################################################
        
#         lims_list.append([min_v, max_v])
#         idx_list_.append(idx_x_)
#         idx_list_m.append(idx_x_m)  
# #     all_masks = torch.from_numpy(all_masks)  #changed
#     if return_mask:
#         return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, all_masks, mask_fft
# ####################


def ifft_compute_color(magnitude, phase, idx, center=False, val_lims=None):
    min_v, max_v = val_lims
    _, h, w = magnitude.shape
    recon_im = np.zeros((3,h,w))
    # magnitude, phase is 3 x h x w
    # idx has 3 elements
    for i in range(3):
        recon_mag = magnitude.copy()[i]
        recon_mag = (max_v - min_v) * recon_mag + min_v
        idx_i = idx[i]
        recon_mag[idx_i] = -np.inf

        recon_phase = phase[i] * (2 * np.pi)

        real_part = np.cos(recon_phase) * np.exp(recon_mag)
        imag_part = np.sin(recon_phase) * np.exp(recon_mag)

        true_fft = np.concatenate((real_part[:, :, None], imag_part[:, :, None]), axis=-1)
        if center:
            true_fft = np.fft.fftshift(true_fft)
        img_back = cv2.idft(true_fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        recon_im[i]=img_back
    return recon_im

def get_color_images_back(x_fft, lims_list, idx_list):
    # x_fft has shape N x 6 x h x w
    h, w = x_fft[0, 0].shape
    N = len(x_fft)
    x_back = np.zeros((N, 3, h, w))
    for i in range(N):
        mag_ = x_fft[i, 0:3]
        phase_ = x_fft[i, 3:]
        idx_ = idx_list[i]
        lims_ = lims_list[i]
        img_back = ifft_compute_color(mag_, phase_, idx_, center=True, val_lims=lims_)
        x_back[i]=img_back
    return x_back

#######################################################################
###############          Color random bbox util         ###############
#######################################################################

from PIL import Image

class FLAGS:
    img_shapes= [64, 64, 3]
    height= 32
    width= 32
    max_delta_height= 8
    max_delta_width= 8
    batch_size= 16
    vertical_margin= 0
    horizontal_margin= 0

def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width

    
    # t = tf.random_uniform([], minval=FLAGS.vertical_margin, maxval=maxt, dtype=tf.int32)
    t = np.random.randint(low=FLAGS.vertical_margin, high=maxt)
    # l = tf.random_uniform([], minval=FLAGS.horizontal_margin, maxval=maxl, dtype=tf.int32)
    l = np.random.randint(low=FLAGS.horizontal_margin, high=maxl)
    h = FLAGS.height
    w = FLAGS.width
    return (t, l, h, w)

def bbox2mask(FLAGS, bbox, name='mask'):
    img_shape = FLAGS.img_shapes
    height = img_shape[0]
    width = img_shape[1]
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    return npmask(bbox, height, width, FLAGS.max_delta_height, FLAGS.max_delta_width)

def product_mask(x, mask):
    return (((x + 1)/2 * mask) - 0.5) / 0.5

 
def make_masked(x_im, flags=FLAGS()):
# x has size N x ch x 64 x 64
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w
    h, w = x_im[0].shape[1:]
    all_masks = np.ones((N, 3, h, w))
            
    for mask_file in range(N):
        bbox = random_bbox(flags)
        mask = bbox2mask(flags, bbox)[0,:,:,0]
        mask = cv2.resize(mask,(w,h))
        mask = mask.copy()

        if len(mask.shape)<=2:
            mask = np.concatenate([mask[...,None]]*3, axis=-1)
            mask = np.transpose(mask, (2,0,1))
            all_masks[mask_file] = 1.0 - mask # 
            
#     import ipdb; ipdb.set_trace()
    x_masked = product_mask(x_im, all_masks)
    x_masked_part = product_mask(x_im, (1. - all_masks))
    return x_masked, x_masked_part, all_masks

def get_color_fft_images_regular(x_im, return_mask=False): # b x 3 x h x w 
    # x_im has size N x ch x 64 x 64
    # x_fft has first three channel as magnitude next 3 channels as phase
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w  
    h, w = x_im[0].shape[1:]
    x_masked, x_masked_part, all_masks = make_masked(x_im)

    x_fft = np.zeros((N, 6, h, w))
    x_masked_fft = np.zeros((N, 6, h, w))
    mask_fft = np.zeros((N, 6, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        x_m = x_masked[i]
        mask = all_masks[i]
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute_color(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute_color(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,:3] = mag_x_
        x_fft[i,3:] = phase_x_
        
        x_masked_fft[i, :3] = mag_x_m
        x_masked_fft[i, 3:] = phase_x_m
        
        
        ##############################################################################
        _, mag_fft_mask_, phase_fft_mask_x_, _ = fft_compute_color(mask, center=True) 
        min_v3 = mag_fft_mask_.min()
        max_v3 = mag_fft_mask_.max()
        mag_fft_mask_ = (mag_fft_mask_ - min_v3)/(max_v3 - min_v3)
        phase_fft_mask_x_ = phase_fft_mask_x_/(2 * np.pi)
        
        mask_fft[i,:3] = mag_fft_mask_
        mask_fft[i,3:] = phase_fft_mask_x_
        ###############################################################################
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)
    all_masks = torch.from_numpy(all_masks)
    if return_mask:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, all_masks, mask_fft
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m


#######################################################################
###############          Color random bbox util         ###############
#######################################################################

from PIL import Image

class FLAGS:
    img_shapes= [64, 64, 3]
    height= 32
    width= 32
    max_delta_height= 8
    max_delta_width= 8
    batch_size= 16
    vertical_margin= 0
    horizontal_margin= 0

def random_bbox(FLAGS):
    """Generate a random tlhw.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = FLAGS.img_shapes
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - FLAGS.vertical_margin - FLAGS.height
    maxl = img_width - FLAGS.horizontal_margin - FLAGS.width

    
    # t = tf.random_uniform([], minval=FLAGS.vertical_margin, maxval=maxt, dtype=tf.int32)
    t = np.random.randint(low=FLAGS.vertical_margin, high=maxt)
    # l = tf.random_uniform([], minval=FLAGS.horizontal_margin, maxval=maxl, dtype=tf.int32)
    l = np.random.randint(low=FLAGS.horizontal_margin, high=maxl)
    h = FLAGS.height
    w = FLAGS.width
    return (t, l, h, w)

def bbox2mask(FLAGS, bbox, name='mask'):
    img_shape = FLAGS.img_shapes
    height = img_shape[0]
    width = img_shape[1]
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    return npmask(bbox, height, width, FLAGS.max_delta_height, FLAGS.max_delta_width)

def product_mask(x, mask):
    return (((x + 1)/2 * mask) - 0.5) / 0.5

 
def make_masked(x_im, flags=FLAGS()):
# x has size N x ch x 64 x 64
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w
    h, w = x_im[0].shape[1:]
    all_masks = np.ones((N, 3, h, w))
            
    for mask_file in range(N):
        bbox = random_bbox(flags)
        mask = bbox2mask(flags, bbox)[0,:,:,0]
        mask = cv2.resize(mask,(w,h))
        mask = mask.copy()

        if len(mask.shape)<=2:
            mask = np.concatenate([mask[...,None]]*3, axis=-1)
            mask = np.transpose(mask, (2,0,1))
            all_masks[mask_file] = 1.0 - mask # 
            
#     import ipdb; ipdb.set_trace()
    x_masked = product_mask(x_im, all_masks)
    x_masked_part = product_mask(x_im, (1. - all_masks))
    return x_masked, x_masked_part, all_masks

def get_color_fft_images_regular(x_im, return_mask=False): # b x 3 x h x w 
    # x_im has size N x ch x 64 x 64
    # x_fft has first three channel as magnitude next 3 channels as phase
    N = len(x_im)
    assert len(x_im[0].shape)==3, "Image should be color" #pytorch 3 x h x w  
    h, w = x_im[0].shape[1:]
    x_masked, x_masked_part, all_masks = make_masked(x_im)

    x_fft = np.zeros((N, 6, h, w))
    x_masked_fft = np.zeros((N, 6, h, w))
    mask_fft = np.zeros((N, 6, h, w))
    
    lims_list = []
    idx_list_ = []
    idx_list_m = []

    for i in range(N):
        x_ = x_im[i]
        x_m = x_masked[i]
        mask = all_masks[i]
        
        _, mag_x_, phase_x_, idx_x_ = fft_compute_color(x_, center=True) 
        min_v1 = mag_x_.min()
        max_v1 = mag_x_.max()
        
        _, mag_x_m, phase_x_m, idx_x_m = fft_compute_color(x_m, center=True) 
        min_v2 = mag_x_m.min()
        max_v2 = mag_x_m.max()
        
        min_v = min(min_v1, min_v2)
        max_v = max(max_v1, max_v2)
        
        mag_x_ = (mag_x_ - min_v)/(max_v - min_v)
        phase_x_ = phase_x_/(2 * np.pi)

        mag_x_m = (mag_x_m - min_v)/(max_v - min_v)
        phase_x_m = phase_x_m/(2 * np.pi)
        
        x_fft[i,:3] = mag_x_
        x_fft[i,3:] = phase_x_
        
        x_masked_fft[i, :3] = mag_x_m
        x_masked_fft[i, 3:] = phase_x_m
        
        
        ##############################################################################
        _, mag_fft_mask_, phase_fft_mask_x_, _ = fft_compute_color(mask, center=True) 
        min_v3 = mag_fft_mask_.min()
        max_v3 = mag_fft_mask_.max()
        mag_fft_mask_ = (mag_fft_mask_ - min_v3)/(max_v3 - min_v3)
        phase_fft_mask_x_ = phase_fft_mask_x_/(2 * np.pi)
        
        mask_fft[i,:3] = mag_fft_mask_
        mask_fft[i,3:] = phase_fft_mask_x_
        ###############################################################################
        
        lims_list.append([min_v, max_v])
        idx_list_.append(idx_x_)
        idx_list_m.append(idx_x_m)
    all_masks = torch.from_numpy(all_masks)
    if return_mask:
        return x_masked,x_masked_part, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m, all_masks, mask_fft
    else:
        return x_masked, x_fft, x_masked_fft, lims_list, idx_list_, idx_list_m
