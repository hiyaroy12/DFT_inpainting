## DFT_inpainting
Image inpainting using frequency domain priors.

## Abstract
In this paper, we present a novel image inpainting technique using frequency domain information. Prior works on image inpainting predict the missing pixels by training neural networks using only the spatial domain information. However, these methods still struggle to reconstruct high-frequency details for real complex scenes, leading to a discrepancy in color, boundary artifacts, distorted patterns, and blurry textures. To alleviate these problems, we investigate if it is possible to obtain better performance by training the networks using frequency domain information (Discrete Fourier Transform) along with the spatial domain information. To this end, we propose a frequency-based deconvolution module that enables the network to learn the global context while selectively reconstructing the high-frequency components. We evaluate our proposed method on the publicly available datasets CelebA, Paris Streetview, and DTD texture dataset, and show that our method outperforms current state-of-the-art image inpainting techniques both qualitatively and quantitatively. 

## Prerequisites: 
- Python 3
- PyTorch 1.0
- NVIDIA GPU + CUDA cuDNN
- some dependencies like cv2, numpy etc. 

## Installation:
- Clone this repo:


## Dataset: 
- Dataset can be downloaded here:

### 1) Training:
To train the model, create a config.yaml file similar to the example config file and copy it under CEEC directory.
- You can train the model for different clusters "n" (0-4 in our case) by using:
```bash
python train.py --dataset mars_hirise --cluster "n" --l1_adv 
```

### 2) Testing
- To test the model for different clusters "n" (0-4 in our case) use:
```bash
python test.py --dataset mars_hirise --cluster "0" --l1_adv
```

### 3) Evaluation:
To evaluate the model, first run the model in test mode against your validation set and save the results on disk. 
Then run metrics.py to evaluate the model using PSNR, SSIM and Mean Absolute Error:
```bash
python metrics.py --data-path [path to validation set] --output-path [path to model output]
```
