import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from utils.config import parse_args
from dataloader.load_avmnist_data import avmnist_load_data
from model.avmnist_image_model import AVMNIST_VisionModel
from utils.metric import count_acc
from utils.function_tools import collate_fn_truncate_image
from dataloader.avmnist_dataloader import AVMNIST

def main(args):
    
    
    model_name = "facebook/wav2vec2-large" if args.mj_model == 'large' else "facebook/wav2vec2-base-960h"
    encoder_dim = 1024 if args.mj_model == 'large' else 768
    
    audio_encoder = Wav2Vec2Model.from_pretrained(model_name, mask_time_prob=0.0, cache_dir="/nfs2/jjlee/model_cache").cuda().eval()
    vision_model = AVMNIST_VisionModel(encoder_dim=encoder_dim).cuda()
    
    optimizer = {
        'adam': torch.optim.Adam,
        'sgd': lambda p, lr: torch.optim.SGD(p, lr, momentum=0.9, weight_decay=5e-4),
        'adamw': torch.optim.AdamW
    }[args.optim](vision_model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
        
    train_data, val_data, test_data = avmnist_load_data(args.data_root, is_spectrogram=False, is_PCA=args.pca)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    datasets = {
        'train': AVMNIST(*train_data, transform = transform, is_spectrogram=False, is_train=True, is_PCA=args.pca),
        'val': AVMNIST(*val_data, transform = transform, is_spectrogram=False, is_train=False, is_PCA=args.pca),
        'test': AVMNIST(*test_data, transform = transform, is_spectrogram=False, is_train=False, is_PCA=args.pca)
    }
    
    dataloaders = {key: DataLoader(dataset, batch_size=args.batch_size, shuffle=(key=='train'),
                                   num_workers=8, pin_memory=True, collate_fn=collate_fn_truncate_image,
                                   drop_last=(key=='train')) for key, dataset in datasets.items()}
    
    best_acc = 0.
    
    print(len(datasets['train']), len(datasets['val']), len(datasets['test']))
    
    for epoch in range(args.epochs):
        vision_model.train()
        train_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        
        for audio, image, label in dataloaders['train']:
            image, audio, label = image.cuda(), audio.cuda(), label.cuda()
            audio_features = audio_encoder(audio).last_hidden_state.mean(dim=1)
            logits, vision_features = vision_model(image)
            
            latent_loss = F.mse_loss(audio_features, vision_features)
            cls_loss = F.cross_entropy(logits, label)
            loss = cls_loss * (1 - args.lam) + (latent_loss * args.lam)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_metrics['cls_loss'] += cls_loss.item()
            train_metrics['latent_loss'] += latent_loss.item()
            train_metrics['acc'] += count_acc(logits, label)
        
        train_metrics = {k: v / len(dataloaders['train']) for k, v in train_metrics.items()}
        print(f"{epoch+1}/{args.epochs} Train - cls: {train_metrics['cls_loss']:.4f}, latent: {train_metrics['latent_loss']:.4f}, acc: {train_metrics['acc']*100:.2f}")
        
        vision_model.eval()
        val_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        with torch.no_grad():
            for audio, image, label in dataloaders['val']:
                image, audio, label = image.cuda(), audio.cuda(), label.cuda()
                audio_features = audio_encoder(audio).last_hidden_state.mean(dim=1)
                logits, vision_features = vision_model(image)
                
                latent_loss = F.mse_loss(audio_features, vision_features)
                cls_loss = F.cross_entropy(logits, label)
                loss = cls_loss * (1 - args.lam) + (latent_loss * args.lam)
                
                val_metrics['cls_loss'] += cls_loss.item()
                val_metrics['latent_loss'] += latent_loss.item()
                val_metrics['acc'] += count_acc(logits, label)
        
        val_metrics = {k: v / len(dataloaders['val']) for k, v in val_metrics.items()}
        print(f"{epoch+1}/{args.epochs} Val - cls: {val_metrics['cls_loss']:.4f}, latent: {val_metrics['latent_loss']:.4f}, acc: {val_metrics['acc']*100:.2f}")
        
        scheduler.step()
        
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            torch.save(vision_model.state_dict(), 'ckpt/avmnist_vision_model_best.pth')
            print(f"{epoch+1}/{args.epochs}: Save best model")
        
    vision_model.load_state_dict(torch.load('ckpt/avmnist_vision_model_best.pth'))
    test_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
    with torch.no_grad():
        for audio, image, label in dataloaders['test']:
            image, audio, label = image.cuda(), audio.cuda(), label.cuda()
            audio_features = audio_encoder(audio).last_hidden_state.mean(dim=1)
            logits, vision_features = vision_model(image)
            
            latent_loss = F.mse_loss(audio_features, vision_features)
            cls_loss = F.cross_entropy(logits, label)
            loss = cls_loss * (1 - args.lam) + (latent_loss * args.lam)
            
            test_metrics['cls_loss'] += cls_loss.item()
            test_metrics['latent_loss'] += latent_loss.item()
            test_metrics['acc'] += count_acc(logits, label)
    
    test_metrics = {k: v / len(dataloaders['test']) for k, v in test_metrics.items()}
    print(f"Test - cls: {test_metrics['cls_loss']:.4f}, latent: {test_metrics['latent_loss']:.4f}, acc: {test_metrics['acc']*100:.2f}")
    return test_metrics['acc']

if __name__ == '__main__':
    
    args = parse_args()
    overall_test_acc = 0.
    for version in range(args.n_times):
        
        test_acc = main(args)
        overall_test_acc += test_acc / args.n_times
    print(f"Overall test acc: {overall_test_acc*100:.2f} AVMNIST A->V ")
