"""
L1+adv+FFT: 
1.stage 1 loads random bbox pretrained models from "../DnCNN-PyTorch/logs" .. e.g. regular_celeba_net.pth
2.stage 2 trains the network for inpainting

CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv_fft-irregular-test.py --dataset dtd --use_irregular 1 --perc_test_mask 0.1
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.1
CUDA_VISIBLE_DEVICES=0 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.1
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.2
CUDA_VISIBLE_DEVICES=2 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.3
CUDA_VISIBLE_DEVICES=3 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.4
CUDA_VISIBLE_DEVICES=4 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.5
CUDA_VISIBLE_DEVICES=5 python CEEC/L1_adv_fft-irregular-test.py --dataset paris_streetview --use_irregular 1 --perc_test_mask 0.6
"""


import os
import sys
import glob
import cv2
import numpy as np
import tqdm
import math
import argparse
import PIL
from PIL import Image
sys.path.append( '../DnCNN+EC' )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import utils
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader, ImageDataset
from CEEC.metrics import PSNR

from utils import my_transform, read_file, MyCelebA, MyDTD, MyParis_streetview, MyPlaces2, Myolivetti
from utils import weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave, Progbar
from freq_utils import make_one_masked_image, get_color_images_back

# from freq_utils import read_file, product_mask, make_masked, random_bbox, get_color_images_back, get_color_fft_images, get_color_fft_images_regular, get_color_fft_images_irregular
from tensorboardX import SummaryWriter

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--dataset", type=str, default="celeba", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--stage1_outf", type=str, default="../TIP_experiments/logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--use_irregular", type=int, default=0, help='When irregular mask is used')
parser.add_argument("--perc_test_mask", type=float, default=0.1, help='percentage irregular mask used for testing')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument("--config_file", type=str, default=None, help='Location of config file')
opt = parser.parse_args()

# config= Config(opt.config_file)
config = Config('./CEEC/config_la_adv_fft.yml')
print(opt)

from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_irregular = bool(opt.use_irregular)
prefix = ('irregular_' if use_irregular else '')

os.makedirs("L1_adv_fft_infer_results/{}{}_images/{}_perc_test_mask".format(prefix, opt.dataset,opt.perc_test_mask), exist_ok=True)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

os.makedirs("L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/clean".format(opt.dataset,opt.perc_test_mask), exist_ok=True)
newdir_clean = "L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/clean".format(opt.dataset,opt.perc_test_mask)
new_root_clean = "/home3/hiya/workspace/inpainting_fft/DnCNN+EC/{}".format(newdir_clean)

os.makedirs("L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/masked".format(opt.dataset,opt.perc_test_mask), exist_ok=True)
newdir_masked = "L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/masked".format(opt.dataset,opt.perc_test_mask)
new_root_masked = "/home3/hiya/workspace/inpainting_fft/DnCNN+EC/{}".format(newdir_masked)

os.makedirs("L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/recon".format(opt.dataset,opt.perc_test_mask), exist_ok=True)
newdir_recon = "L1_adv_fft_test_results/irregular/{}_images/{}_perc_test_mask/recon".format(opt.dataset,opt.perc_test_mask)
new_root_recon = "/home3/hiya/workspace/inpainting_fft/DnCNN+EC/{}".format(newdir_recon)

# def save_each_image(x_in, logdir, dirname, step=0):
#     N = len(x_in)
#     rootdir = os.path.join(logdir, dirname)
#     os.makedirs(rootdir, exist_ok=True)
#     for k in range(N):
#         utils.save_image(x_in[k], rootdir + '/step_{}_image_{}.png'.format(step, k))


# PSNR metric
# psnr_compute = PSNR(255.0).to(config.DEVICE)

# ----------
#  Load stage1
# ----------

def load_stage1_model():
    net_stage1 = DnCNN(channels=12, out_ch=6, num_of_layers=opt.num_of_layers)
    if use_irregular:
        PATH = '{}/irregular_{}_net.pth'.format(opt.stage1_outf, opt.dataset)
    print('Loading from ', PATH)
    ckpt = torch.load(PATH, map_location='cpu')
    net_stage1.load_state_dict(ckpt)
    return net_stage1

net_stage1 = load_stage1_model().to(device)

# loader_train, test_loader = get_data_loader(opt)
model = InpaintingModel(config, gen_in_channel=9).to(device)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Initialize generator weights
if use_irregular:
    PATH = "L1_adv_fft_results/{}{}_generator.h5f".format(prefix, opt.dataset)
# else:
#     PATH = "CEEC_fft_results/{}_generator.h5f".format(opt.dataset)
print('Loading from ', PATH)
ckpt = torch.load(PATH, map_location='cpu')
model.generator.load_state_dict(ckpt)

# Initialize discriminator weights
if use_irregular:
    PATH = "L1_adv_fft_results/{}{}_discriminator.h5f".format(prefix, opt.dataset)
# else:
#     PATH = "CEEC_fft_results/{}_discriminator.h5f".format(opt.dataset)
print('Loading from ', PATH)
ckpt = torch.load(PATH, map_location='cpu')
model.discriminator.load_state_dict(ckpt)

# ----------
#  Inference
# ----------
model.eval()
# psnr_list = []
# mae_list = []

# for i, (imgs, _) in enumerate(test_loader):     
        
#     ############ Mask the images and compute the FFTs ##########################################################
#     if use_irregular:
#         outputs_irregular = get_color_fft_images_irregular(imgs.numpy(), True, False, opt.perc_test_mask)
#         x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = outputs_irregular
imfiles = sorted(glob.glob('/home3/hiya/workspace/inpainting_fft/data/{}_test/clean/*'.format(opt.dataset)))
maskfiles = sorted(glob.glob('/home3/hiya/workspace/inpainting_fft/data/{}_test/irregular_masks/{}percent/*'.format(opt.dataset,opt.perc_test_mask)))
transform=transforms.Compose([transforms.Resize((64, 64)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                      (0.5, 0.5, 0.5))])

for imfile, maskfile in zip(imfiles, maskfiles):
    img = Image.open(imfile)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    mask = cv2.imread(maskfile, 0)
    mask = mask[None,...]
    mask = mask.copy()/255.
    mask = np.concatenate([mask[...,None]]*3, axis=-1)
#     mask = torch.tensor(1.0 - mask)
    
    outs = make_one_masked_image(batch_t, mask, True)
    
    x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = outs

    masked_imgs = torch.from_numpy(x_masked).type(Tensor) 
    masks = torch.tensor(np.transpose(all_masks, [0,3,1,2])).type(Tensor)
#     masks = all_masks.type(Tensor) 
#     masks = torch.tensor(np.transpose(masks.detach().cpu().numpy(), [0,3,1,2]))
    imgs = batch_t.type(Tensor)  
#     masks = torch.from_numpy(all_masks)
#     masks = np.transpose(masks,(0,3,1,2))
#     masks = masks.type(Tensor) 
#   masks = all_masks.type(Tensor)                              
#     imgs = imgs.type(Tensor) 
    
    
    img_train = torch.from_numpy(x_fft).type(Tensor)                                #changed (x_fft~6)
    imgn_train = torch.from_numpy(x_masked_fft).type(Tensor)                        #changed (x_masked_fft~6)
    img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda()) #added   (img_train~6,imgn_train~6)
        
    mask_fft = torch.from_numpy(mask_fft).type(torch.FloatTensor)                   #added   (mask_fft~6)
    mask_train_in = Variable(mask_fft.cuda())                                       #added   (mask_train_in~6)
        
    imgn_train_cat = torch.cat((mask_train_in, imgn_train), axis=1)  #added  (imgn_train_cat~12:mask_train_in~6,imgn_train~6)
        
    out_train = torch.clamp(net_stage1(imgn_train_cat), 0., 1.)             #(out_train~12)      
#      out_train = torch.clamp(net_stage1(imgn_train_1), 0., 1.)

    img_back = get_color_images_back(img_train.cpu().numpy(), lims_list, idx_list)             #added orig img back~3
    img_back_masked = get_color_images_back(imgn_train.cpu().numpy(), lims_list, idx_list_m)   #added masked img back~3
    img_back_recon = get_color_images_back(out_train.detach().cpu().numpy(), lims_list, idx_list) #recon img back~3
    img_back_recon = torch.clamp(torch.from_numpy(img_back_recon), -1., 1.).type(Tensor)
        
#   sample = torch.cat((masked_imgs, img_back_recon.data, imgs.data), -2)
#   save_image(sample, "test.png", nrow=8, normalize=True)
#   import ipdb; ipdb.set_trace()
        
    masked_imgs_display = masked_imgs.clone()
    masked_imgs = torch.cat((masked_imgs, img_back_recon), axis=1) #masked_imgs=masked_imgs+ifft of 1st_stage
        
    i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
    outputs_merged = (i_outputs * (1 - masks)) + (imgs * masks)
    
    basename = os.path.basename(imfile)
    
    newname_clean = os.path.join(new_root_clean, basename)
    save_image((imgs.data + 1.)/2, newname_clean)
    
    newname_masked = os.path.join(new_root_masked, basename)
    save_image((masked_imgs_display.data + 1.)/2, newname_masked)
    
    newname_recon = os.path.join(new_root_recon, basename)
    save_image((outputs_merged.data + 1.)/2, newname_recon)

    ########################################################################################################

#     # metrics
#     psnr = psnr_compute(postprocess((imgs+1.)/2.), postprocess((outputs_merged+1.)/2.))
#     mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
        
#     i_logs.append(('psnr', psnr.item()))
#     i_logs.append(('mae', mae.item()))
    
#     psnr_list.append(i_logs[5][1])
#     mae_list.append(i_logs[6][1])
    
#     for log in i_logs:
#         print(log[0]+' : '+str(log[1]))
        
#     # Generate sample at sample interval
#     sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
#     save_image(sample, "L1_adv_fft_infer_results/{}{}_images/{}_perc_test_mask/%d.png".format(prefix, opt.dataset, opt.perc_test_mask) % i, nrow=8, normalize=True)
    
#     # Save sample
#     step=i
              
#     save_each_image((imgs.data + 1)/2, "L1_adv_fft_infer_results/irregular/{}_images/{}_perc_test_mask".format(opt.dataset, opt.perc_test_mask), 'clean', step=step)
#     save_each_image((masked_imgs_display.data + 1)/2, "L1_adv_fft_infer_results/irregular/{}_images/{}_perc_test_mask".format(opt.dataset, opt.perc_test_mask), 'noisy', step=step)
#     save_each_image((img_back_recon.data + 1)/2, "L1_adv_fft_infer_results/irregular/{}_images/{}_perc_test_mask".format(opt.dataset, opt.perc_test_mask), 'stage1_recon', step=step)
#     save_each_image((outputs_merged.data + 1)/2, "L1_adv_fft_infer_results/irregular/{}_images/{}_perc_test_mask".format(opt.dataset, opt.perc_test_mask), 'reconstructed', step=step)
    

# print('Avg PSNR: ', np.mean(psnr_list))
# print('Avg MAE: ', np.mean(mae_list))
    
    
    
    
    
    
    
#         batches_done = epoch * len(loader_train) + i
#         if batches_done % opt.sample_interval == 0:
#             sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
#             save_image(sample, "L1_adv_fft_results/{}{}_images/%d.png".format(prefix, opt.dataset) % batches_done, nrow=8, normalize=True)
    
#     torch.save(model.generator.state_dict(),"L1_adv_fft_results/{}{}_generator.h5f".format(prefix, opt.dataset))
#     torch.save(model.discriminator.state_dict(),"L1_adv_fft_results/{}{}_discriminator.h5f".format(prefix, opt.dataset))
     
    
    
