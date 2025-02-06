# Vision-Language Models (VLMs)

This is the official PyTorch implementation of our paper on Vision-Language Models (VLMs). Our implementation focuses on the following architectures:

- **Vision Models**: ResNet50, ViT-B/32, ViT-B/16
- **Language Models**: BERT, RoBERTa

## Download Datasets

We train Vision-Language Models on ImageNet and evaluate them on robustness benchmarks. Download the required datasets:

- **ImageNet Dataset**: Available from the official [ImageNet Website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
- **Evaluation Benchmarks**: Execute the provided script: [download_data.sh](script/download_data.sh)

## Training

Our hyperparameter settings follow the implementations from two key papers:
- [DAT](https://arxiv.org/abs/2209.07735) ([Github Code](https://github.com/alibaba/easyrobust/tree/main/examples/imageclassification/imagenet/dat))
- [DAD](https://arxiv.org/abs/2311.01441) ([Github Code](https://github.com/lapisrocks/DiscreteAdversarialDistillation))

To train the models using 4 GPUs, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_adaptive.py \
--data_dir=/your/own/path/ImageNet \
--test_dir=/your/own/path/ImageNet_OOD \
--model=vit_base_patch16_224 \
--language_model=bert \
--workers=8 \
--batch-size=256 \
--epochs=300 \
--weight-decay=5e-2 \
--lr=1e-3 \
--lam=0.5 \
--drop-path=0.25 \
--model-ema \
--model-ema-decay=0.99992 \
--opt-eps=1e-8 \
--opt=adamw \
--sched=cosine \
--warmup-lr=1e-6 \
--warmup-epochs=20 \
--cooldown-epochs=10 \
--patience-epochs=10 \
--aa=rand-m9-mstd0.5-inc1 \
--reprob=0.25 \
--smoothing=0.1 \
--mixup=0.8 \
--cutmix=1.0 \
--pin-mem \
--mean 0.485 0.456 0.406 \
--std 0.229 0.224 0.225 \
--output=../output \
--experiment=vit_b16_bert_large \
--log-interval=100 \
--clip-grad=1.0 \
--amp
```

*Note: Adjust the `--output` directory path to specify where checkpoints should be saved.*

Additional training scripts can be found in the [script](script/) directory. 🚀