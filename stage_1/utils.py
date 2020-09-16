import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from torchvision.datasets.celeba import CelebA
import PIL
from PIL import Image
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets.celeba import CelebA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state


#for cityscapes
def my_transform(x, y):
    transform1=transforms.Compose([transforms.Resize((256, 256)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5), 
                                                                                      (0.5, 0.5, 0.5))])
    
    transform2=transforms.Compose([transforms.Resize((256, 256)),
                                                                 transforms.ToTensor(),])
    return transform1(x), transform2(y)

def read_file(filename):
    return [line.rstrip('\n') for line in open(filename)]

class MyCelebA(CelebA):
    def __init__(self, *args, **kwargs): #main file that needs to be modified
        super().__init__( *args, **kwargs) 
        self.filename = [x.replace('.jpg', '.png') for x in self.filename]
        
    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        
        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))
                
        h,w,c = np.array(X).shape
        #frac = 0.2
        frac = 0.0
        x1, y1, w, h = frac * w, frac * h, (1-2*frac)*w, (1-2*frac)*h
        X = X.crop((x1, y1, x1+w, y1+h))
        #X = X.crop((y1, x1, y1+h, x1+w))
        
        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
                    
        return X, target


class MyDTD(CelebA):
    def __init__(self, *args, **kwargs): #main file that needs to be modified
        super().__init__( *args, **kwargs) 
        self.filename = [x.replace('.jpg', '.png') for x in self.filename]
        if self.split == 'train':
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/DnCNN-PyTorch/data/dtd/dtd_train_files.txt')
        else:
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/DnCNN-PyTorch/data/dtd/dtd_test_files.txt')
 
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.filenames[index])
        h,w,c = np.array(X).shape
        frac = 0.2
        x1, y1, w, h = frac * w, frac * h, (1-2*frac)*w, (1-2*frac)*h
        X = X.crop((x1, y1, x1+w, y1+h))
        #X = X.crop((y1, x1, y1+h, x1+w))
        if self.transform is not None:
            X = self.transform(X)
        return X, X
    
class MyParis_streetview(CelebA):
    def __init__(self, *args, **kwargs): #main file that needs to be modified
        super().__init__( *args, **kwargs) 
        self.filename = [x.replace('.JPG','.png') for x in self.filename]
        if self.split == 'train':
#             self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/psv/paris_JPG_train_images.txt')
             self.filenames =read_file('/home3/hiya/workspace/inpainting_fft/does128work/data/paris_JPG_train_images.txt')
        else:
#             self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/psv/paris_png_val_images.txt')
            self.filenames =read_file('/home3/hiya/workspace/inpainting_fft/does128work/data/paris_png_val_images.txt')
 
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.filenames[index])
        h,w,c = np.array(X).shape
        frac = 0.0
        x1, y1, w, h = frac * w, frac * h, (1-2*frac)*w, (1-2*frac)*h
        X = X.crop((x1, y1, x1+w, y1+h))
        #X = X.crop((y1, x1, y1+h, x1+w))
        if self.transform is not None:
            X = self.transform(X)
        return X, X
    
class MyPlaces2(CelebA):
    def __init__(self, *args, **kwargs): #main file that needs to be modified
        super().__init__( *args, **kwargs) 
        self.filename = [x.replace('.jpg','.png') for x in self.filename]
        if self.split == 'train':
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/places2/places2_train_files.txt')
        else:
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/places2/places2_val_files.txt')
 
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.filenames[index])
        np_x = np.array(X)
        if len(np_x.shape)<=2:
            print('Found grayscale image')
            np_x = np.concatenate([np_x[...,None]*3], axis=1)
            X = X.convert('RGB')
        try:
            
            h,w,c = np_x.shape
        except:
            import ipdb;ipdb.set_trace()
        frac = 0.0
        x1, y1, w, h = frac * w, frac * h, (1-2*frac)*w, (1-2*frac)*h
        X = X.crop((x1, y1, x1+w, y1+h))
        #X = X.crop((y1, x1, y1+h, x1+w))
        if self.transform is not None:
            X = self.transform(X)
        return X, X
    

class Myolivetti(CelebA):
    def __init__(self, *args, **kwargs): #main file that needs to be modified
        super().__init__( *args, **kwargs) 
        self.filename = [x.replace('.jpg', '.png') for x in self.filename]
        self.datas = fetch_olivetti_faces()['images']
        
        if self.split == 'train':
            self.x_data = self.datas
        else:
            self.x_data = self.datas[-50:]
 
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        X = Image.fromarray(self.x_data[index])
        #X = X.crop((y1, x1, y1+h, x1+w))
        if self.transform is not None:
            X = self.transform(X)
        return X, X
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


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
