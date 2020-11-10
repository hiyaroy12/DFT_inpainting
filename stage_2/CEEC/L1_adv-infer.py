"""
L1+Adv Inference
Inpainting with Edge-connect inpainting model with L1+Adv

CUDA_VISIBLE_DEVICES=8 python CEEC/L1_adv-infer.py --dataset svhn --use_regular 1
CUDA_VISIBLE_DEVICES=8 python CEEC/L1_adv-infer.py --dataset dtd --use_regular 1
CUDA_VISIBLE_DEVICES=5 python CEEC/L1_adv-infer.py --dataset celeba --use_regular 1
CUDA_VISIBLE_DEVICES=8 python CEEC/L1_adv-infer.py --dataset paris_streetview --use_regular 1

"""


import os, sys
import numpy as np
np.random.seed(1234)
import math, PIL
import argparse
from PIL import Image
sys.path.append( '../DnCNN+EC' )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader, ImageDataset
from CEEC.metrics import PSNR

from utils import my_transform, read_file, MyCelebA, MyDTD, MyParis_streetview, MyPlaces2, Myolivetti
from utils import weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave, Progbar

from freq_utils import product_mask, make_masked, random_bbox, get_color_images_back, get_color_fft_images, get_color_fft_images_regular
from tensorboardX import SummaryWriter

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=120, help="size of the batches")
parser.add_argument("--dataset", type=str, default="celeba", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
# parser.add_argument("--stage1_outf", type=str, default="../DnCNN-PyTorch/logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--use_regular", type=int, default=0, help='When random_bbox for regular mask is used')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument("--config_file", type=str, default=None, help='Location of config file')
opt = parser.parse_args()

# config= Config(opt.config_file)
config = Config('./CEEC/config_la_adv_fft.yml')

print(opt)

from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_regular = bool(opt.use_regular)
prefix = ('random_bbox_' if use_regular else '')

os.makedirs("L1_Adv_infer_results/{}{}_images".format(prefix, opt.dataset), exist_ok=True)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
# use_cuda = torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

# PSNR metric
psnr_compute = PSNR(255.0).to(config.DEVICE)

def save_each_image(x_in, logdir, dirname, step=0):
    N = len(x_in)
    rootdir = os.path.join(logdir, dirname)
    os.makedirs(rootdir, exist_ok=True)
    for k in range(N):
        utils.save_image(x_in[k], rootdir + '/step_{}_image_{}.png'.format(step, k))

loader_train, test_loader = get_data_loader(opt)
model = InpaintingModel(config, gen_in_channel=6).to(device)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Initialize generator weights
if use_regular:
    PATH = "L1_Adv_results/{}{}_generator.h5f".format(prefix, opt.dataset)
else:
    PATH = "L1_Adv_results/{}_generator.h5f".format(opt.dataset)
print('Loading from ', PATH)
ckpt = torch.load(PATH, map_location='cpu')
model.generator.load_state_dict(ckpt)

# Initialize discriminator weights
if use_regular:
    PATH = "L1_Adv_results/{}{}_discriminator.h5f".format(prefix, opt.dataset)
else:
    PATH = "L1_Adv_results/{}_discriminator.h5f".format(opt.dataset)
print('Loading from ', PATH)
ckpt = torch.load(PATH, map_location='cpu')
model.discriminator.load_state_dict(ckpt)

# ----------
# Inference
# ----------
model.eval()
psnr_list = []
mae_list = []

for i, (imgs, _) in enumerate(test_loader):     
        
    # Mask the images and compute the FFTs also
    if use_regular:
        outputs_regular = get_color_fft_images_regular(imgs.numpy(), True)
        x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = outputs_regular

    masked_imgs = torch.from_numpy(x_masked).type(Tensor) 
    masks = all_masks.type(Tensor) 
    imgs = imgs.type(Tensor) 

    masked_imgs_display = masked_imgs.clone()
#     masked_imgs = torch.cat((masked_imgs, img_back_recon), axis=1)
        
    i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
    outputs_merged = (i_outputs * (1 - masks)) + (imgs * masks)

    # metrics
    psnr = psnr_compute(postprocess((imgs+1.)/2.), postprocess((outputs_merged+1.)/2.))
    mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
        
    i_logs.append(('psnr', psnr.item()))
    i_logs.append(('mae', mae.item()))
        
    psnr_list.append(i_logs[5][1])
    mae_list.append(i_logs[6][1])
        
    for log in i_logs:
        print(log[0]+' : '+str(log[1]))

    # Generate sample at sample interval
    sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
    save_image(sample, "L1_Adv_infer_results/{}{}_images/%d.png".format(prefix, opt.dataset) % i, nrow=8, normalize=True)
    
    # Save sample
    step=i

    save_each_image((imgs.data + 1)/2, "L1_Adv_infer_results/random_bbox/{}_images".format(opt.dataset), 'clean', step=step)
    save_each_image((masked_imgs_display.data + 1)/2, "L1_Adv_infer_results/random_bbox/{}_images".format(opt.dataset), 'noisy', step=step)
    save_each_image((outputs_merged.data + 1)/2, "L1_Adv_infer_results/random_bbox/{}_images".format(opt.dataset), 'reconstructed', step=step)
    

print('Avg PSNR: ', np.mean(psnr_list))
print('Avg MAE: ', np.mean(mae_list))

