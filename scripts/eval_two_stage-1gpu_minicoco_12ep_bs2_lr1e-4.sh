set -e
coco_path=data/coco
num_gpus=1
num_enc_layers=6
num_dec_layers=6
dataset=minicoco
batch_size=2
device_code=homeworkstation
backbone=resnet18
num_queries=300
branch=enc_o2m_assign_v1.0

exp_code=${branch}-shortersize_480-${device_code}-${dataset}-msdetr_baseline-${backbone}_enc${num_enc_layers}_dec${num_dec_layers}_query${num_queries}-bs${batch_size}x${num_gpus}_lr1e-4
EXP_DIR=exps/${exp_code}

mkdir -p $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --backbone $backbone \
   --two_stage \
   --batch_size $batch_size \
   --wandb_name ${exp_code} \
   --wandb_enabled \
   --lr 1e-4 \
   --lr_backbone 1e-5 \
   --enc_layers $num_enc_layers \
   --dec_layers $num_dec_layers \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=$coco_path \
   --num_queries $num_queries \
   --dropout 0.0 \
   --mixed_selection \
   --look_forward_twice \
   --use_ms_detr \
   --use_aux_ffn \
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 2 \
   --enc_cls_loss_coef 2 \
   --enc_bbox_loss_coef 5 \
   --enc_giou_loss_coef 2 \
   --resume exps/enc_o2m_assign_v1.0-shortersize_480-homeworkstation-minicoco-msdetr_baseline-resnet18_enc6_dec6_query300-bs2x1_lr1e-4/checkpoint.pth \
   --eval
   