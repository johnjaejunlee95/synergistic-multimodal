import torch 
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from utils.metric import *


class IEMOCAP_LanguageModel(torch.nn.Module):
    def __init__(self, hidden_size=768):
        super(IEMOCAP_LanguageModel, self).__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="/nfs2/jjlee/model_cache", add_prefix_space=True)
        self.encoder = BertModel.from_pretrained("bert-base-uncased", cache_dir="/nfs2/jjlee/model_cache").cuda()
        
        
        self.linear_transform = torch.nn.Sequential(*[
            torch.nn.Linear(self.encoder.config.hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
        ])
                
        self.fc = nn.Linear(hidden_size, 6) 

        
    def forward(self, x):

        text_tokenized = self.tokenizer(x, max_length = 128 , add_special_tokens=True, truncation=True, padding=True, return_tensors="pt").to("cuda") 
        text_embeddings = self.encoder(**text_tokenized)
        text_embeddings = text_embeddings['last_hidden_state'][:,0,:]
        features = self.linear_transform(text_embeddings)
        
        output = self.fc(features)
        return output, features 
