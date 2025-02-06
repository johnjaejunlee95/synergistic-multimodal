# Multimodal Models

This is the official PyTorch implementation of our paper on multimodal models. Our implementation focuses on the following architectures:

## Model Architectures 

> Note: In our notation, A → B means model A helps model B, where A is the pretrained model.

### IEMOCAP Dataset
- **Language → Audio**
  - Language: BERT-base/large (pretrained)
  - Audio: Wav2Vec2-base
- **Audio → Language**
  - Audio: Wav2Vec2-base/large (pretrained)
  - Language: BERT-base

### AVMNIST Dataset
- **Vision → Audio**
  - Vision: ResNet-18/34 (pretrained)
  - Audio: Audio Model (ResNet type)
- **Audio → Vision**
  - Audio: Wav2Vec2-base/large (pretrained)
  - Vision: ResNet-18

## Datasets

We train our models on IEMOCAP and AVMNIST datasets:

- **IEMOCAP**: Available from the [official website](https://sail.usc.edu/iemocap/iemocap_release.htm)
  - Requires submission of access request
  - Dataset size: approximately 16GB

- **AVMNIST**: Available via [Google Drive](https://drive.google.com/file/d/1oXkkioKMOErndhvdu8PKGstIMCFSHhdA/view?usp=sharing)
  - Download and extract the archive

## Training Commands

### AVMNIST Audio → Vision
```bash
python AVMNIST_A2V.py \
  --epochs 30 \
  --learning_rate 1e-3 \
  --batch_size 32 \
  --data_root /your/own/avmnist_path \
  --optim adam \
  --lam 0.3 \
  --lr_decay_step 25 \
  --lr_decay_ratio 0.1 \
  --mj_model resnet18 \
  --n_times 3 \
  --random_seed 42
```

### AVMNIST Vision → Audio
```bash
python AVMNIST_V2A.py \
  --epochs 30 \
  --learning_rate 1e-3 \
  --batch_size 32 \
  --data_root /your/own/avmnist_path \
  --optim adam \
  --lam 0.3 \
  --lr_decay_step 25 \
  --lr_decay_ratio 0.1 \
  --mj_model resnet18 \
  --n_times 3 \
  --random_seed 42
```

### IEMOCAP Audio → Text
```bash
python IEMOCAP_A2T.py \
  --epochs 30 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --optim adam \
  --lam 0.3 \
  --audio_dir /your/own/IEMOCAP_path \
  --text_dir /your/own/IEMOCAP_path/IEMOCAP/texts/ \
  --mj_model base \
  --n_times 3 \
  --random_seed 1234
```

### IEMOCAP Text → Audio
```bash
python IEMOCAP_T2A.py \
  --epochs 30 \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --optim adam \
  --lam 0.3 \
  --audio_dir /your/own/IEMOCAP_path \
  --text_dir /your/own/IEMOCAP_path/IEMOCAP/texts/ \
  --language_model bert \
  --n_times 3 \
  --random_seed 1234
```