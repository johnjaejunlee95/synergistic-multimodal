CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=34912 ../test_adaptive.py \
--data_dir=/dataset/ImageNet \
--test_dir=/dataset/ImageNet_OOD \
--model=vit_base_patch32_224 \
--language_model=bert \
--workers=8 \
--batch-size=512 \
--drop-path=0.25 \
--model-ema \
--pin-mem \
--mean 0.485 0.456 0.406  \
--std 0.229 0.224 0.225 \
--output=../output \
--experiment=vit_b32_bert_large \
--log-interval=100 \
--clip-grad=1.0 \
--resume=/nfs1/jjlee/ckpt/imagenet/vit_b32_roberta_large/model_best.pth.tar \

# all file paths need to be changed into your own paths
