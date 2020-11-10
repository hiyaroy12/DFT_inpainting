#bash CEEC/metric_cal/metrics_all.sh

#L1_adv_fft
########################################################################################################
echo "metrics for L1_Adv_FFT_dtd"
#need to do for different percentage of test mask
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.1_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.2_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.2_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.3_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.4_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.5_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.6_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/dtd_images/0.6_perc_test_mask/reconstructed/

########################################################################################################
echo "metrics for L1_Adv_FFT_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/reconstructed/

########################################################################################################
echo "metrics for L1_Adv_FFT_celeba"
CUDA_VISIBLE_DEVICES=9 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/celeba_images/clean/ --output-path ./L1_adv_fft_infer_results/irregular/celeba_images/reconstructed/

CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/irregular/celeba_images/0.1_perc_test_mask/clean/ --output-path ./L1_adv_fft_infer_results/irregular/celeba_images/0.1_perc_test_mask/reconstructed/ 


CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/paris_streetview_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/paris_streetview_images/reconstructed/


CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/clean/ --output-path ./L1_adv_fft_infer_results/random_bbox/celeba_images/reconstructed/

CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path 











########################################################################################################
########################################################################################################

#CEEC_only
#need to do for different percentage of test mask
echo "metrics for L1_Adv_S_P_dtd"

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.2_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/dtd_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/dtd_images/0.6_perc_test_mask/reconstructed/

########################################################################################################
echo "metrics for L1_Adv_S_P_celeba"
CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.1_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.2_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=5 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/celeba_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/celeba_images/0.6_perc_test_mask/reconstructed/


########################################################################################################
echo "metrics for L1_Adv_S_P_PSV"

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_only_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/reconstructed/

########################################################################################################
########################################################################################################

#CEEC_fft
echo "metrics for L1_Adv_S_P_FFT_dtd"
CUDA_VISIBLE_DEVICES=9 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.2_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/dtd_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/dtd_images/0.6_perc_test_mask/reconstructed/



########################################################################################################

echo "metrics for L1_Adv_S_P_FFT_celeba"
CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.1_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.2_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.4_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.5_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=2 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/celeba_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/celeba_images/0.6_perc_test_mask/reconstructed/


########################################################################################################

echo "metrics for L1_Adv_S_P_FFT_PSV"
CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.1_perc_test_mask/reconstructed/


CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.2_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.3_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.4_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.5_perc_test_mask/reconstructed/

CUDA_VISIBLE_DEVICES=8 python CEEC/metric_cal/metrics.py --data-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/clean/ --output-path ./CEEC_fft_infer_results/irregular/paris_streetview_images/0.6_perc_test_mask/reconstructed/
########################################################################################################

 