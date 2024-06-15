set -e
coco_path=data/coco
num_gpus=4
num_enc_layers=6
num_dec_layers=6
dataset=minicoco
batch_size=4
device_code=3090x8_1
branch=sparse_token_v3.0.3_ratio_0.3
backbone=resnet18
num_queries=100

exp_code=no_two_stage-shortersize480-${device_code}_${dataset}-${branch}_${backbone}_enc${num_enc_layers}_dec${num_dec_layers}_query${num_queries}-bs${batch_size}x$num_gpus
EXP_DIR=exps/${exp_code}

mkdir -p $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --backbone $backbone \
   --wandb_enabled \
   --wandb_name ${exp_code} \
   --batch_size $batch_size \
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
   2>&1 | tee $EXP_DIR/train.log
