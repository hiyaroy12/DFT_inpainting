import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
from torchvision.datasets.celeba import CelebA
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, time, random
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
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/psv/paris_JPG_train_images.txt')
        else:
            self.filenames = read_file('/home3/hiya/workspace/inpainting_fft/TIP_experiments/data/psv/paris_png_val_images.txt')
 
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
        return X, self.filenames[index]
    
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

#########################extra utils from Edge-connect###################################

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
