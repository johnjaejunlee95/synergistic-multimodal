import os 
import torch
import torch.nn.functional as F
import numpy as np
from utils.config import parse_args
from dataloader.iemocap_dataloader import IEMOCAP
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from model.iemocap_audio_model import IEMOCAP_AudioModel
from utils.metric import count_acc
from utils.function_tools import collate_fn_padding
from torch.utils.data import DataLoader, random_split

def main(args):
    
    
    if args.language_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/nfs2/jjlee/model_cache", add_prefix_space=False)
        language_model = BertModel.from_pretrained('bert-base-uncased', add_pooling_layer=False, cache_dir="/nfs2/jjlee/model_cache").cuda()
    elif args.language_model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir="/nfs2/jjlee/model_cache", add_prefix_space=False)
        language_model = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, cache_dir="/nfs2/jjlee/model_cache").cuda()
    
    
    audio_model = IEMOCAP_AudioModel(language_model).cuda()
    
    optimizer = {
        'adam': lambda p: torch.optim.Adam(p, lr=args.learning_rate, weight_decay=5e-4),
        'sgd': lambda p: torch.optim.SGD(p, lr=args.learning_rate, momentum=0.9),
        'adamw': lambda p: torch.optim.AdamW(p, lr=args.learning_rate, weight_decay=5e-4)
    }[args.optim](audio_model.parameters())
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 2, num_training_steps = 100000000)
    
    datasets = IEMOCAP(args.audio_dir, args.text_dir)
    train_size = int(len(datasets) * 0.8)
    val_size = int(len(datasets) * 0.1)
    test_size = len(datasets) - train_size - val_size
    train_datasets, val_datasets, test_datasets = random_split(datasets, [train_size, val_size, test_size])
    
    dataloaders = {
        'train': DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn_padding, drop_last=True),
        'val': DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_padding),
        'test': DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_padding)
    }
    
    best_acc = 0.
    for epoch in range(args.epochs):
        audio_model.train()
        train_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        
        for padded_audio, text, label in dataloaders['train']:
            audio, ratio = padded_audio
            audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
            
            
            
            logits, audio_features = audio_model(audio, ratio)
            text_tokenized = tokenizer(text, padding=True, return_tensors="pt").to("cuda")
            text_embeddings = language_model(**text_tokenized)["last_hidden_state"].mean(dim=1)
            
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
        
        audio_model.eval()
        val_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
        with torch.no_grad():
            for padded_audio, text, label in dataloaders['val']:
                audio, ratio = padded_audio
                audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
                
                logits, audio_features = audio_model(audio, ratio)
                text_tokenized = tokenizer(text, padding=True, return_tensors="pt").to("cuda")
                text_embeddings = language_model(**text_tokenized)["last_hidden_state"].mean(dim=1)
                
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
            torch.save(audio_model.state_dict(), f'ckpt/iemocap_audio_model_best.pth')
            print(f"{epoch+1}/{args.epochs}: Save best model")
    
    
    audio_model.load_state_dict(torch.load('ckpt/iemocap_audio_model_best.pth'))
    test_metrics = {'cls_loss': 0., 'latent_loss': 0., 'acc': 0.}
    with torch.no_grad():
        for padded_audio, text, label in dataloaders['test']:
            audio, ratio = padded_audio
            audio, ratio, label = audio.cuda(), ratio.cuda(), label.cuda()
                        
            logits, audio_features = audio_model(audio, ratio)
            text_tokenized = tokenizer(text, padding=True, return_tensors="pt").to("cuda")
            text_embeddings = language_model(**text_tokenized)["last_hidden_state"].mean(dim=1)
            
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

    print(f"Overall test acc: {overall_test_acc*100:.2f} IEMOCAP T->A")
