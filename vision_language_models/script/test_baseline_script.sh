CUDA_VISIBLE_DEVICES=0,1,2 taskset --cpu-list 10-18,30-38,50-58,70-78 torchrun --nproc_per_node=3 --master_port 1234 ../test_baseline.py \
--prompts_dir=/nfs2/jjlee/code/VLM-OOD/prompts \
--data_dir=/dataset/ImageNet \
--test_dir=/dataset/ImageNet_OOD \
--model=resnet50 \
--workers=8 \
--batch-size=512 \
--drop-path=0.25 \
--model-ema \
--pin-mem \
--mean 0.485 0.456 0.406  \
--std 0.229 0.224 0.225 \
--output=../output \
--experiment=vit_b32 \
--log-interval=100 \
--clip-grad=1.0 \
--resume=model_best.pth.tar \

# all file paths need to be changed into your own paths