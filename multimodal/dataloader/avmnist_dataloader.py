import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random 
import torch.nn.functional as F
import torchaudio
from PIL import Image 
import torchvision.transforms as transforms
# import librosa
# import scipy.signal
# import scipy.io.wavfile

class AVMNIST(Dataset):
    def __init__(self, image, audio, label, transform=None, is_spectrogram=False, is_train=False, is_PCA=True) -> None:
        super(AVMNIST,self).__init__()
        
        
        self.image = image
        self.audio = audio
        self.label = label
        
        self.length = len(self.image)
        self.transform = transform
        self.is_spectrogram = is_spectrogram
        self.is_train = is_train
        self.is_pca = is_PCA

    

    def __getitem__(self, index):
        image = self.image[index]
        audio = self.audio[index]
        label = self.label[index]
        label = torch.tensor(label).long()

        random_idx = random.choice([i for i in range((self.length)) if i != index])
        
        if self.is_spectrogram:
            audio = torch.from_numpy(audio).reshape(1, 112, 112) / 255.0
            
            if self.is_train:
                image = self.image[random_idx] 
                
            if self.is_pca:
                image = torch.from_numpy(image).reshape(1, 28, 28) / 255.0
            else: # Original MNIST image
                image = Image.fromarray(image[0])
                if self.transform:
                    image = self.transform(image)
                image = transforms.Grayscale(num_output_channels=1)(image)  # Ensure grayscale
            image = image.repeat(3, 1, 1)
                        
            image = image.float()
            audio = audio.float()
            
            return audio, image, label
        
        
        else:
            if self.is_pca:
                image = torch.from_numpy(image).reshape(1, 28, 28) / 255.0
            else:
                image = Image.fromarray(image[0])
                if self.transform:
                    image = self.transform(image)
            
            if self.is_train:
                audio = self.audio[random_idx]
                waveform, sample_rate = torchaudio.load(audio)
                waveform = waveform.view(1, -1)
                wavefrom += torch.randn_like(waveform) * 1e-3
            else:
                waveform, sample_rate = torchaudio.load(audio)
                waveform = waveform.view(1, -1)
                wavefrom += torch.randn_like(waveform) * 1e-3

            image = image.float()
            waveform = waveform.float()
            return waveform, image, label
    
    def __len__(self):
        return self.length
    