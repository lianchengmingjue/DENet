K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res

INPUT_SIZE=256
DATASET=LOGO
NAME=slbr

python3  test_lpips.py \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --dataset_dir  ./data/LOGO/10khigh\
  --preprocess resize \
  --no_flip \
  --name ${NAME} \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --dataset ${DATASET} \
  --resume ./pretrain/logo_high_model_best.pth.tar \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP}
  
    # --checkpoint /media/sda/Watermark \
  
