set -e
coco_path=data/coco
num_gpus=1
num_enc_layers=3
num_dec_layers=3
dataset=minicoco
batch_size=4
branch_name="cascade_msdetr"
code_version="repvgg"

EXP_DIR=exps/${dataset}-${branch_name}_${code_version}-cascade_msdetr_enc${num_enc_layers}_dec${num_dec_layers}-bs${batch_size}

mkdir -p $EXP_DIR
cp -r models $EXP_DIR

GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python -u main.py \
   --lr 5e-5 \
   --lr_backbone 5e-6 \
   --batch_size $batch_size \
   --enc_layers $num_enc_layers \
   --dec_layers $num_dec_layers \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --coco_path=$coco_path \
   --num_queries 300 \
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


