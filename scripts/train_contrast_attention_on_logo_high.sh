K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res #'cat'

L1_LOSS=2
CONTENT_LOSS=2.5e-1
STYLE_LOSS=2.5e-1
PRIMARY_LOSS=0.01
IOU_LOSS=0.25 
CONTRAST_LOSS=0.25

INPUT_SIZE=256
DATASET=LOGO
NAME=SLBR_CONTRAST_Attention_LOGO_HIGH
LOGNAME=sample_Contrast_Attention_2Unet_LOGO_High.log
# nohup python -u   main.py \
python -u train.py \
 --epochs 200 \
 --schedule 80 \
 --lr 1e-3 \
 --gpu_id 1,2,3,4 \
 --checkpoint ./weight \
 --dataset_dir ./data/LOGO/10khigh \
 --nets slbr  \
 --sltype vggx \
 --mask_mode ${MASK_MODE} \
 --lambda_content ${CONTENT_LOSS} \
 --lambda_style ${STYLE_LOSS} \
 --lambda_iou ${IOU_LOSS} \
 --lambda_l1 ${L1_LOSS} \
 --lambda_primary ${PRIMARY_LOSS} \
 --lambda_contrast ${CONTRAST_LOSS} \
 --masked True \
 --loss-type hybrid \
 --models slbr \
 --input-size ${INPUT_SIZE} \
 --crop_size ${INPUT_SIZE} \
 --train-batch 16 \
 --test-batch 1 \
 --preprocess resize \
 --name ${NAME} \
 --k_center ${K_CENTER} \
 --dataset ${DATASET} \
 --use_refine \
 --k_refine ${K_REFINE} \
 --k_skip_stage ${K_SKIP} \
 --log ${LOGNAME} \
