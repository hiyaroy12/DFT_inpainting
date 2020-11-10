"""
Inpainting with Edge-connect inpainting model only


The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
CUDA_VISIBLE_DEVICES=3 python CEEC/L1_adv-irregular.py --dataset paris_streetview --use_irregular 1
CUDA_VISIBLE_DEVICES=4 python CEEC/L1_adv-irregular.py --dataset celeba --use_irregular 1
CUDA_VISIBLE_DEVICES=5 python CEEC/L1_adv-irregular.py --dataset dtd --use_irregular 1
"""

import os, sys
import numpy as np
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
from torchvision import datasets, transforms
from torchvision.utils import save_image
from CEEC.models import InpaintingModel, DnCNN
from CEEC.config import Config
from CEEC.dataloader import get_data_loader, ImageDataset
from CEEC.metrics import PSNR

from utils import my_transform, read_file, MyCelebA, MyDTD, MyParis_streetview, MyPlaces2, Myolivetti
from utils import weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave, Progbar

from freq_utils import read_file, product_mask, make_masked, random_bbox, get_color_images_back, get_color_fft_images, get_color_fft_images_regular, get_color_fft_images_irregular
from tensorboardX import SummaryWriter

def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=210, help="size of the batches")
parser.add_argument("--dataset", type=str, default="celeba", help="name of the dataset")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
# parser.add_argument("--stage1_outf", type=str, default="../DnCNN-PyTorch/logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--use_irregular", type=int, default=0, help='When irregular mask is used')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
opt = parser.parse_args()
print(opt)

# config = Config('./CEEC/config.yml')
config = Config('./CEEC/config_la_adv_fft.yml')


from datetime import datetime
now = datetime.now() # current date and time
datetime_f = "_".join(str(now).split())

use_irregular = bool(opt.use_irregular)
prefix = ('irregular_' if use_irregular else '')

os.makedirs("L1_Adv_results/{}{}_images".format(prefix, opt.dataset), exist_ok=True)
use_cuda = not opt.no_cuda and torch.cuda.is_available()
torch.manual_seed(1234)
device = torch.device("cuda" if use_cuda else "cpu")

# PSNR metric
psnr_compute = PSNR(255.0).to(config.DEVICE)

#  Dataloader
loader_train, test_loader = get_data_loader(opt)

# ----------
#  Training
# ----------
model = InpaintingModel(config, gen_in_channel=6).to(device)
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(loader_train):     

        if use_irregular:
            outputs_irregular = get_color_fft_images_irregular(imgs.numpy(), True, True, None)
            x_masked, x_fft, x_masked_fft, lims_list, idx_list, idx_list_m, all_masks, mask_fft = outputs_irregular

        masked_imgs = torch.from_numpy(x_masked).type(Tensor) 
        masks = torch.from_numpy(all_masks)
        masks = np.transpose(masks,(0,3,1,2))
        masks = masks.type(Tensor)                             
        imgs = imgs.type(Tensor) 
        
        masked_imgs_display = masked_imgs.clone()
        
        i_outputs, i_gen_loss, i_dis_loss, i_logs = model.process(imgs, masked_imgs, masks)
        outputs_merged = (i_outputs * (1 - masks)) + (imgs * masks)

        # metrics
        psnr = psnr_compute(postprocess((imgs+1.)/2.), postprocess((outputs_merged+1.)/2.))
        mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
        
        i_logs.append(('psnr', psnr.item()))
        i_logs.append(('mae', mae.item()))
        
        print("[Epoch %d/%d] [Batch %d/%d]"% (epoch, opt.n_epochs, i, len(loader_train)))
        for log in i_logs:
            print(log[0]+' : '+str(log[1]))
            
        # backward
        model.backward(i_gen_loss, i_dis_loss)
        iteration = model.iteration
              
#         import ipdb; ipdb.set_trace()
              
        # Generate sample at sample interval
        batches_done = epoch * len(loader_train) + i
        if batches_done % opt.sample_interval == 0:
            sample = torch.cat((masked_imgs_display.data, outputs_merged.data, imgs.data), -2)
            save_image(sample, "L1_Adv_results/{}{}_images/%d.png".format(prefix, opt.dataset) % batches_done, nrow=8, normalize=True)
    
    torch.save(model.generator.state_dict(),"L1_Adv_results/{}{}_generator.h5f".format(prefix, opt.dataset))
    torch.save(model.discriminator.state_dict(),"L1_Adv_results/{}{}_discriminator.h5f".format(prefix, opt.dataset))
     
    
    
