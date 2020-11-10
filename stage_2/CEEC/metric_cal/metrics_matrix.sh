#bash CEEC/metric_cal/metrics_all.sh
#for table 
#################################################################
#patchmatch #psv #irregular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/clean/ --output-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/irregular_patchmatch_inpainted/0.1percent/

#patchmatch #psv #regular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/clean --output-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/rrbox_patchmatch_inpainted/
#################################################################
#Context encoder #psv #irregular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_test_results/irregular/paris_streetview_images/0.6_perc_test_mask/clean/ --output-path ./L1_adv_test_results/irregular/paris_streetview_images/0.6_perc_test_mask/recon/

#Context encoder #psv #regular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_test_results/random_bbox/paris_streetview_images/clean --output-path ./L1_adv_test_results/random_bbox/paris_streetview_images/recon
#################################################################
#deepfill(CA) #psv #irregular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/clean/ --output-path /home3/hiya/workspace/inpainting_fft/deepfill_v1/our_results/dset_paris_streetview/irregular/0.2per/

#deepfill(CA) #psv #regular(done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/paris_streetview_test/clean/ --output-path /home3/hiya/workspace/inpainting_fft/deepfill_v1/our_results/dset_paris_streetview/
#################################################################
#ours #psv #irregular (check)

#ours #psv #regular (done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_test_results/random_bbox/paris_streetview_images/clean --output-path ./L1_adv_fft_test_results/random_bbox/paris_streetview_images/recon
#################################################################
#################################################################
#patchmatch #celeba #irregular (done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/celeba_test/clean/ --output-path /home3/hiya/workspace/inpainting_fft/data/celeba_test/irregular_patchmatch_inpainted/0.1percent/

#patchmatch #celeba #regular (done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path /home3/hiya/workspace/inpainting_fft/data/celeba_test/clean --output-path /home3/hiya/workspace/inpainting_fft/data/celeba_test/rrbox_patchmatch_inpainted/
#################################################################
#Context encoder #celeba #irregular (done)
CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_test_results/irregular/celeba_images/0.2_perc_test_mask/clean/ --output-path ./L1_adv_test_results/irregular/celeba_images/0.2_perc_test_mask/recon/

#Context encoder #celeba #regular (done)
CUDA_VISIBLE_DEVICES=4 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_test_results/random_bbox/celeba_images/clean/ --output-path ./L1_adv_test_results/random_bbox/celeba_images/recon/
#################################################################
#deepfill(CA) #celeba #irregular



#deepfill(CA) #celeba #regular

#################################################################
#ours #celeba #irregular (check)



#ours #celeba #regular (done)
CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_test_results/random_bbox/celeba_images/clean --output-path ./L1_adv_fft_test_results/random_bbox/celeba_images/recon




#1 ours #psv #irregular (check)
#2 deepfill(CA) #celeba #irregular (do)
#3 deepfill(CA) #celeba #regular (do)
#4 ours #celeba #irregular (check)





CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./L1_adv_fft_test_results/random_bbox/dtd_images/clean --output-path ./L1_adv_fft_test_results/random_bbox/dtd_images/recon



CUDA_VISIBLE_DEVICES=0 python CEEC/metric_cal/metrics.py --data-path ./L1_only_test_results/random_bbox/paris_streetview_images/clean --output-path ./L1_only_test_results/random_bbox/paris_streetview_images/recon