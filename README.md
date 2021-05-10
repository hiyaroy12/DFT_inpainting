## DFT_inpainting
Image inpainting using frequency domain priors.

## Abstract
In this paper, we present a novel image inpainting technique using frequency domain information. Prior works on image inpainting predict the missing pixels by training neural networks using only the spatial domain information. However, these methods still struggle to reconstruct high-frequency details for real complex scenes, leading to a discrepancy in color, boundary artifacts, distorted patterns, and blurry textures. To alleviate these problems, we investigate if it is possible to obtain better performance by training the networks using frequency domain information (Discrete Fourier Transform) along with the spatial domain information. To this end, we propose a frequency-based deconvolution module that enables the network to learn the global context while selectively reconstructing the high-frequency components. We evaluate our proposed method on the publicly available datasets CelebA, Paris Streetview, and DTD texture dataset, and show that our method outperforms current state-of-the-art image inpainting techniques both qualitatively and quantitatively. 

## Prerequisites: 
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN
- some dependencies like cv2, numpy etc. 


## Installation
- Clone this repo:
```bash
git clone https://github.com/hiyaroy12/DFT_inpainting.git
cd DFT_inpainting
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
### 1) Images
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder) and [DTD texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/) datasets. You can download the datasets from the official websites to train the model. 

### 2) Irregular Masks
We train our model on the irregular mask dataset provided by [Yu et al.]() You can download the Irregular Mask Dataset from [their website]().

We test our model on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download the Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

## Getting Started
You can download the pre-trained models from the following links and keep them under `./checkpoints` directory.

[CelebA]() | [Paris StreetView]() | [DTD texture dataset]()

### 1) Training
Our model is trained in two stages: 1) training the deconvolution module and 2) training the refinement model. 
#### Train the deconvolution module using:
- Train the model for `regular mask` using:
```bash
CUDA_VISIBLE_DEVICES=1 python stage_1/train_color-randombbox.py --epochs 100 --dataset celeba --use_regular 1
```

- Train the model for `irregular mask` using:
```bash
CUDA_VISIBLE_DEVICES=1 python stage_1/train_color_irregular.py --epochs 100 --dataset celeba --use_irregular 1
```

#### Train the refinement module using:
- Train the model for `regular mask` using:
```bash
python stage_2/CEEC/L1_adv_fft.py --n_epochs [] --dataset [] --use_regular 1
CUDA_VISIBLE_DEVICES=1 python stage_2/CEEC/L1_adv_fft.py --dataset celeba --n_epochs 300 --use_regular 1 (Example)
```
Example:
```bash
CUDA_VISIBLE_DEVICES=1 python stage_2/CEEC/L1_adv_fft.py --dataset celeba --n_epochs 300 --use_regular 1
```

- Train the model for `irregular mask` using:
```bash
python stage_2/CEEC/L1_adv_fft-irregular.py --n_epochs [] --dataset [] --use_irregular 1
```
Example:
```bash
CUDA_VISIBLE_DEVICES=1 python stage_2/CEEC/L1_adv-irregular.py --n_epochs 300 --dataset celeba --use_irregular 1
```

### 2) Testing
To test the model:

- Please download the [stage-1 pre-trained models](https://drive.google.com/drive/folders/1ZWtyd8jb9R14OqJN0ytpgYNDz3XbLu7_?usp=sharing) for CelebA, Paris StreetView, and DTD datasets, put them into `logs/` (Please check the model path correctly in the code). Here `regular_{}_net.pth` and `irregular_{}_net.pth` refer to regular and irregular masks.

- Please download the [stage-2 pre-trained models](https://drive.google.com/drive/folders/1K4ry5qlkzMzk3ZqrS1sLm4p949ebIXfv?usp=sharing) for CelebA, Paris StreetView, and DTD datasets, put them into `L1_adv_fft_results/`. Here `random_bbox_{}_generator.h5f`, `random_bbox_{}_discriminator.h5f` refer to regular masks and `irregular_{}_generator.h5f`, `irregular_{}_discriminator.h5f` refer to irregular masks.

- Then for testing against your validation set for regular masks, run:
```bash
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv_fft-test.py --dataset [dataset_name] --use_regular 1
```
Example:
```bash
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv_fft-test.py --dataset celeba --use_regular 1
```

- Testing against your validation set for irregular masks, run:
```bash
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv-irregular-test.py --dataset [dataset_name] --use_irregular 1 --perc_test_mask []
```
Example:
```bash
CUDA_VISIBLE_DEVICES=1 python CEEC/L1_adv-irregular-test.py --dataset celeba --use_irregular 1 --perc_test_mask 0.1
```

### 3) Evaluating
To evaluate the model, first run the model in test mode against your validation set and save the results on disk. 

#### Pre-trained Models: 
- Please download the [stage-1 pre-trained models](https://drive.google.com/drive/folders/1ZWtyd8jb9R14OqJN0ytpgYNDz3XbLu7_?usp=sharing) for CelebA, Paris StreetView, and DTD datasets, put them into `logs/` (Please check the model path correctly in the code). Here `regular_{}_net.pth` and `irregular_{}_net.pth` refer to regular and irregular masks.

- Please download the [stage-2 pre-trained models](https://drive.google.com/drive/folders/1K4ry5qlkzMzk3ZqrS1sLm4p949ebIXfv?usp=sharing) for CelebA, Paris StreetView, and DTD datasets, put them into `L1_adv_fft_results/`. Here `random_bbox_{}_generator.h5f`, `random_bbox_{}_discriminator.h5f` refer to regular masks and `irregular_{}_generator.h5f`, `irregular_{}_discriminator.h5f` refer to irregular masks.

#### Metric calculation:
Then run metrics.py to evaluate the model using PSNR, SSIM and Mean Absolute Error:
```bash
CUDA_VISIBLE_DEVICES=9 python CEEC/metric_cal/metrics.py --data-path [path to validation set] --output-path [path to model output]
```
Example: 
```bash
CUDA_VISIBLE_DEVICES=9 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/dtd_images/clean/ --output-path ./CEEC_fft_infer_results/dtd_images/reconstructed/ 
```


