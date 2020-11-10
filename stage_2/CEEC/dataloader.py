# # Dataset loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.celeba import CelebA
import torchvision.transforms as transforms
from utils import my_transform, read_file, MyCelebA, MyDTD, MyParis_streetview, MyPlaces2, Myolivetti, weights_init_kaiming, batch_PSNR, data_augmentation, create_dir, create_mask, stitch_images, imshow, imsave, Progbar

import glob
import random
import os
import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()

def get_data_loader(opt):
    print('Loading dataset ...\n')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if opt.dataset=='celeba':
        dataset_train = MyCelebA('../data', split="train", target_type="bbox", download=False,
                                       transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                          (0.5, 0.5, 0.5))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = MyCelebA('../data', split="valid",  target_type="bbox",
                                                transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                  (0.5, 0.5, 0.5))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)

    elif opt.dataset=='svhn':
        dataset_train = datasets.SVHN('../data/SVHN', split='train', download=True, target_transform=None,
                                              transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                          (0.5, 0.5, 0.5))]))
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)


        dataset_val = datasets.SVHN('../data/SVHN', split='test', download=True, target_transform=None,
                                              transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                          (0.5, 0.5, 0.5))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)

    elif opt.dataset=='dtd':
        dataset_train = MyDTD('../data', split="train", download=False,
                                       transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                          (0.5, 0.5, 0.5))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = MyDTD('../data', split="valid", 
                                       transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                  (0.5, 0.5, 0.5))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)

    elif opt.dataset=='olivetti':
        dataset_train = Myolivetti('../data', split="train", download=False,
                                           transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.5,), 
                                                                                              (0.5,)), 
                                                                         lambda x: x.repeat( 3, 1, 1)]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = Myolivetti('../data', split="valid", 
                                            transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.5,),
                                                                                              (0.5,)), 
                                                                          lambda x: x.repeat( 3, 1, 1)]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)


    elif opt.dataset=='paris_streetview':
        dataset_train = MyParis_streetview('../data', split="train", download=False,
                                       transform=transforms.Compose([transforms.Resize((opt.img_size,opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                          (0.5, 0.5, 0.5))]))

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True)

        dataset_val = MyParis_streetview('../data', split="valid", 
                                        transform=transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))]))
        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True)


    print("# of training samples: %d\n" % int(len(dataset_train)))
#     print("# of testing samples: %d\n" % int(len(test_loader)))
    print("# of testing samples: %d\n" % int(len(dataset_val)))
    
    
    return loader_train, test_loader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            # For training data perform random mask
            masked_img, aux = self.apply_random_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
