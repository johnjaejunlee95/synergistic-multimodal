import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision 


class AVMNIST_VisionModel(torch.nn.Module):
    def __init__(self, encoder_dim=768):
        super(AVMNIST_VisionModel, self).__init__()
        
        
        self.encoder = torchvision.models.resnet18(weights=None)
        self.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = torch.nn.Linear(512, encoder_dim)
        
        self.fc = torch.nn.Sequential(*[
            torch.nn.LayerNorm(encoder_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(encoder_dim, 10)
        ])
    
        
    def forward(self, x):
        feature = self.encoder(x)
        output = self.fc(feature)
        return output, feature