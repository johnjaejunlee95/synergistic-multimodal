import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.config import parse_args
from dataloader.iemocap_dataloader import IEMOCAP
from model.iemocap_language_model import IEMOCAP_LanguageModel
from transformers import Wav2Vec2Model, get_cosine_schedule_with_warmup
from utils.metric import count_acc
from utils.function_tools import make_padding_masks, collate_fn_padding
from torch.utils.data import DataLoader, random_split


def main(args):
    
    
    model_name = "facebook/wav2vec2-large" if args.mj_model == 'large' else "facebook/wav2vec2-base-960h"
    encoder_dim = 1024 if args.mj_model == 'large' else 768
    
    audio_encoder = Wav2Vec2Model.from_pretrained(model_name, mask_time_prob=0.1, cache_dir="/nfs2/jjlee/model_cache").cuda().eval()
    language_model = IEMOCAP_LanguageModel(encoder_dim).cuda()
    language_model.requires_grad_(True)
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    
    optimizer = {
        'adam': lambda p: torch.optim.Adam(p, lr=args.learning_rate, weight_decay=5e-4),
        'sgd': lambda p: torch.optim.SGD(p, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4),
        'adamw': lambda p: torch.optim.AdamW(p, lr=args.learning_rate, weight_decay=5e-4)
    }[args.optim](language_model.parameters())
    
    datasets = IEMOCAP(args.audio_dir, args.text_dir, is_audio=False)
    train_size = int(len(datasets) * 0.75)
    val_size = int(len(datasets) * 0.1)
    test_size = len(datasets) - train_size - val_size
    train_datasets, val_datasets, test_datasets = random_split(datasets, [train_size, val_size, test_size])
    
    dataloaders = {
        'train': DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn_padding, drop_last=True),
        'val': DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_padding),
        'test': DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_padding)
    }
    
    scheduler =  get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 2, num_training_steps = 100000000)
    best_acc = 0.
    for epoch in range(args.epochs):
        language_model.train()
        train_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        
        for audio, text, label in dataloaders['train']:
            audio, ratio = audio
            audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
            padded_mask = make_padding_masks(audio, ratio)
            
            logits, text_embeddings = language_model(text)
            audio_features = audio_encoder(audio, attention_mask=padded_mask).last_hidden_state.mean(dim=1)
            
            latent_loss = F.mse_loss(audio_features, text_embeddings)
            cls_loss = F.cross_entropy(logits, label)
            loss = cls_loss * (1 - args.lam) + latent_loss * args.lam
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_metrics['cls_loss'] += cls_loss.item()
            train_metrics['latent_loss'] += latent_loss.item()
            train_metrics['acc'] += count_acc(logits, label)
        
        train_metrics = {k: v / len(dataloaders['train']) for k, v in train_metrics.items()}
        print(f"{epoch+1}/{args.epochs} Train - cls: {train_metrics['cls_loss']:.4f}, latent: {train_metrics['latent_loss']:.4f}, acc: {train_metrics['acc']*100:.2f}")
        
        language_model.eval()
        val_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        with torch.no_grad():
            for audio, text, label in dataloaders['val']:
                audio, ratio = audio
                audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
                padded_mask = make_padding_masks(audio, ratio)
                
                logits, text_embeddings = language_model(text)
                audio_features = audio_encoder(audio, attention_mask=padded_mask).last_hidden_state.mean(dim=1)
                
                latent_loss = F.mse_loss(audio_features, text_embeddings)
                cls_loss = F.cross_entropy(logits, label)
                loss = cls_loss * (1 - args.lam) + latent_loss * args.lam
                
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
            torch.save(language_model.state_dict(), 'ckpt/iemocap_text_model_best.pth')
            print(f"{epoch+1}/{args.epochs}: Save best model")
    
    language_model.load_state_dict(torch.load('ckpt/iemocap_text_model_best.pth'))
    test_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
    with torch.no_grad():
        for audio, text, label in dataloaders['test']:
            audio, ratio = audio
            audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
            padded_mask = make_padding_masks(audio, ratio)
            
            logits, text_embeddings = language_model(text)
            audio_features = audio_encoder(audio, attention_mask=padded_mask).last_hidden_state.mean(dim=1)
            
            latent_loss = F.mse_loss(audio_features, text_embeddings)
            cls_loss = F.cross_entropy(logits, label)
            loss = cls_loss * (1 - args.lam) + latent_loss * args.lam
            
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

    print(f"Overall test acc: {overall_test_acc*100:.2f} IEMOCAP A->T")