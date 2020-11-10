#bash CEEC/metric_cal/metrics_all.sh

#L1_only
echo "metrics for L1_only_svhn"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_only_infer_results/random_bbox/svhn_images/clean/ --output-path ./L1_only_infer_results/random_bbox/svhn_images/reconstructed/


echo "metrics for L1_only_dtd"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_only_infer_results/random_bbox/dtd_images/clean/ --output-path ./L1_only_infer_results/random_bbox/dtd_images/reconstructed/

echo "metrics for L1_only_celeba"
CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./L1_only_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_only_infer_results/random_bbox/celeba_images/reconstructed/

echo "metrics for L1_only_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_only_infer_results/random_bbox/paris_streetview_images/clean/ --output-path ./L1_only_infer_results/random_bbox/paris_streetview_images/reconstructed/


#L1_adv
echo "metrics for L1_Adv_SVHN"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_Adv_infer_results/random_bbox/svhn_images/clean/ --output-path ./L1_Adv_infer_results/random_bbox/svhn_images/reconstructed/

echo "metrics for L1_Adv_dtd"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_Adv_infer_results/random_bbox/dtd_images/clean/ --output-path ./L1_Adv_infer_results/random_bbox/dtd_images/reconstructed/


echo "metrics for L1_Adv_celeba"
CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./L1_Adv_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_Adv_infer_results/random_bbox/celeba_images/reconstructed/

echo "metrics for L1_Adv_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_Adv_infer_results/random_bbox/paris_streetview_images/clean/  --output-path ./L1_Adv_infer_results/random_bbox/paris_streetview_images/reconstructed/

########################################################################################################

#L1_adv_fft
echo "metrics for L1_Adv_FFT_SVHN"
CUDA_VISIBLE_DEVICES=7 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/svhn_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/svhn_images/reconstructed/

echo "metrics for L1_Adv_FFT_dtd"
CUDA_VISIBLE_DEVICES=9 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/dtd_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/dtd_images/reconstructed/


echo "metrics for L1_Adv_FFT_celeba"
CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/reconstructed/

echo "metrics for L1_Adv_FFT_PSV"
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/paris_streetview_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/paris_streetview_images/reconstructed/

./L1_adv_fft_infer_results/random_bbox/paris_streetview_images/clean

########################################################################################################

#CEEC_only
echo "metrics for L1_Adv_S_P_SVHN"
CUDA_VISIBLE_DEVICES=7 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/random_bbox/svhn_images/clean/ --output-path ./CEEC_only_infer_results/random_bbox/svhn_images/reconstructed/

echo "metrics for L1_Adv_S_P_dtd"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/random_bbox/dtd_images/clean/ --output-path ./CEEC_only_infer_results/random_bbox/dtd_images/reconstructed/


echo "metrics for L1_Adv_S_P_celeba"
CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/random_bbox/celeba_images/clean/ --output-path ./CEEC_only_infer_results/random_bbox/celeba_images/reconstructed/


echo "metrics for L1_Adv_S_P_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/random_bbox/paris_streetview_images/clean/ --output-path ./CEEC_only_infer_results/random_bbox/paris_streetview_images/reconstructed/

inpainting_fft/DnCNN+EC/CEEC_only_infer_results/random_bbox/paris_streetview_images/reconstructed

########################################################################################################

#CEEC_fft
echo "metrics for L1_Adv_S_P_FFT_SVHN"
CUDA_VISIBLE_DEVICES=1 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/random_bbox/svhn_images/clean/ --output-path ./CEEC_fft_infer_results/random_bbox/svhn_images/reconstructed/

echo "metrics for L1_Adv_S_P_FFT_dtd"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/random_bbox/dtd_images/clean/ --output-path ./CEEC_fft_infer_results/random_bbox/dtd_images/reconstructed/

echo "metrics for L1_Adv_S_P_FFT_celeba"
CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/random_bbox/celeba_images/clean/ --output-path ./CEEC_fft_infer_results/random_bbox/celeba_images/reconstructed/


echo "metrics for L1_Adv_S_P_FFT_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/random_bbox/paris_streetview_images/clean/ --output-path ./CEEC_fft_infer_results/random_bbox/paris_streetview_images/reconstructed/


########################################################################################################

 CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_Adv_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_Adv_infer_results/random_bbox/celeba_images/reconstructed/
 
 CUDA_VISIBLE_DEVICES=7 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/reconstructed/
 
 CUDA_VISIBLE_DEVICES=1 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/random_bbox/celeba_images/clean/ --output-path ./CEEC_only_infer_results/random_bbox/celeba_images/reconstructed/
 
 CUDA_VISIBLE_DEVICES=6 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/random_bbox/celeba_images/clean/ --output-path ./CEEC_fft_infer_results/random_bbox/celeba_images/reconstructed/
 
 
 