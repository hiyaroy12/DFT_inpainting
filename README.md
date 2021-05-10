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
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris Street-View](https://github.com/pathak22/context-encoder) and [DTD texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/) datasets. You can download the datasets from the official websites to train the model. 

### 2) Irregular Masks
We train our model on the irregular mask dataset provided by [ et al.](). You can download the Irregular Mask Dataset from [their website]().

We test our model on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download the Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

## Getting Started
You can download the pre-trained models from the following links and keep them under `./checkpoints` directory.

[CelebA]() | [Paris-StreetView]() | [DTD texture dataset]()

### 1) Training
Our model is trained in two stages: 1) training the deconvolution module and 2) training the refinement model. 
#### Train the deconvolution module using:
```bash
CUDA_VISIBLE_DEVICES=6 python stage_1/train_color-randombbox.py --dataset celeba --use_regular 1
```
#### Train the refinement module using:
Create a `config.yaml` file similar to the [example config file]() and copy it under CEEC directory.
Train the model for "regular mask" using:
```bash
python stage_2/CEEC/L1_adv_fft.py --n_epochs [] --dataset [] --use_regular 1
```
Train the model for "irregular mask" using:
```bash
python stage_2/CEEC/L1_adv_fft-irregular.py --n_epochs [] --dataset [] --use_irregular 1
```

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. 

To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](https://drive.google.com/drive/folders/1K4ry5qlkzMzk3ZqrS1sLm4p949ebIXfv?usp=sharing) for CelebA and Paris-StreetView datasets, put them into `L1_adv_fft_results/`
and run:
```bash
python test.py \
  --checkpoints ./checkpoints/places2 
  --input ./examples/places2/images 
  --mask ./examples/places2/masks
  --output ./checkpoints/results
```
Here `random_bbox_{}_generator.h5f`, `random_bbox_{}_discriminator.h5f` refer to regular masks and `irregular_{}_generator.h5f`, `irregular_{}_discriminator.h5f` refer to irregular masks.

### 3) Evaluating
To evaluate the model,first run the model in test mode against your validation set and save the results on disk. 

#### Pre-trained Models: 
Please download the [pre-trained models](https://drive.google.com/drive/folders/1K4ry5qlkzMzk3ZqrS1sLm4p949ebIXfv?usp=sharing) for CelebA and Paris-StreetView datasets and put them into `L1_adv_fft_results/`

Here `random_bbox_{}_generator.h5f`, `random_bbox_{}_discriminator.h5f` refer to regular masks and `irregular_{}_generator.h5f`, `irregular_{}_discriminator.h5f` refer to irregular masks.

#### Metric calculation:
Then run metrics.py to evaluate the model using PSNR, SSIM and Mean Absolute Error:
```bash
python metrics.py --data-path [path to validation set] --output-path [path to model output]
```


