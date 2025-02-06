import os
from torch.utils.data import Dataset
import torchaudio

import json
import torch 
import random


class IEMOCAP(Dataset):

    def __init__(self, audio_dir, text_dir, is_audio = True):

        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.is_audio = is_audio
        
        self._indices = torchaudio.datasets.IEMOCAP(root=self.audio_dir)
        
  
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        audio, sr, index_name, labels, _ = self._indices[index]
        
        # Label mapping
        label_mapping = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "exc": 4, "fru": 5}
        label = label_mapping[labels]
        
        path = os.path.join(self.text_dir, index_name)
        with open(path + ".json", 'r') as f:
            text = json.load(f)
        language_data = text['Utterance']
        
        if self.is_audio:
            random_class = torch.randint(0, 6, (1,))
            language_data = f"This is about Emotion {random_class}."
            
        else:
            random_idx = random.choice([i for i in range(len(self._indices)) if i != index])
            audio, _, _, _, _ = self._indices[random_idx]
            
        audio += torch.randn_like(audio) * 1e-3
            
        return audio, language_data, label