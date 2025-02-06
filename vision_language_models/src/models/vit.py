import torch
from timm.models import create_model
from src.utils.utils import *



class Vision_Transformers(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.vision_model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=0,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript)
        
        if 'resnet' in args.model:
            self.linear_transform = torch.nn.Linear(self.vision_model.layer4[0].downsample[1].weight.size(0), args.hidden_size)
        else:
            self.linear_transform = torch.nn.Linear(self.vision_model.norm.weight.size(0), args.hidden_size)
        
        
        self.layer_norm = torch.nn.LayerNorm(args.hidden_size)       
        self.linear_classification = torch.nn.Linear(args.hidden_size, args.num_classes)
        

    def forward(self, inputs, is_test=True):
        vision_features = self.vision_model(inputs)
        vision_features = self.linear_transform(vision_features)
        vision_features = self.layer_norm(vision_features)
        
        vision_outputs = self.linear_classification(vision_features)

        if is_test is False:
            return  vision_features, vision_outputs
        return vision_outputs