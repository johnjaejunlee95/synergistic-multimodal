import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from utils.metric import StatisticsPooling
from utils.function_tools import *

class IEMOCAP_AudioModel(nn.Module):
    def __init__(self, text_encoder):
        super(IEMOCAP_AudioModel, self).__init__()
        
        self.avg_pooling = StatisticsPooling()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", mask_time_prob=0.3, cache_dir="/nfs2/jjlee/model_cache").cuda()
        
        self.encoder.freeze_feature_encoder()
        self.encoder.do_normalize = True
        self.encoder.masked_spec_embed.requires_grad = True
        self.linear = nn.Linear(768, text_encoder.config.hidden_size)
        
        self.fc = torch.nn.Sequential(*[
            torch.nn.Linear(text_encoder.config.hidden_size, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 6)
        ])
        
    def forward(self, x, length):
        
        padded_mask = make_padding_masks(x, length)
        
        feature = self.encoder(x, attention_mask=padded_mask).last_hidden_state
        feature - F.layer_norm(feature, feature.size()[1:])
        feature = self.avg_pooling(feature, length).squeeze(1)
        feature = self.linear(feature)
        
        output = self.fc(feature)
        return output, feature