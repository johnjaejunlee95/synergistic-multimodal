import os
import torch
from tqdm import tqdm

from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from timm.utils import AverageMeter, reduce_tensor, accuracy

def evaluate_stylized_imagenet(model, data_dir, test_batchsize=128, test_transform=None, amp_autocast=None, dist=False):
    if not os.path.exists(data_dir):
        print('{} is not exist. skip')
        return

    if dist:
        assert torch.distributed.is_available() and torch.distributed.is_initialized()

    # device = next(model.parameters()).device

    # stylized imagenet always has size of 224
    if test_transform is None:
        # instyle_transform = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), # 256
        #                                         transforms.CenterCrop(224),
        #                                         transforms.ToTensor(), 
        #                                         # transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
        #                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                     ])
        instyle_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        instyle_transform = test_transform

    dataset_instyle = datasets.ImageFolder(data_dir, transform=instyle_transform)
    
    sampler = None
    if dist:
        sampler = torch.utils.data.DistributedSampler(dataset_instyle, shuffle=False)

    instyle_data_loader = torch.utils.data.DataLoader(
                    dataset_instyle, sampler=sampler,
                    batch_size=test_batchsize,
                    num_workers=4,
                    pin_memory=True,
                    drop_last=False
                )
            
    top1_m = AverageMeter()
    # model.eval()
    for input, target in (instyle_data_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            if amp_autocast:
                with torch.cuda.amp.autocast():
                    output = model(input)
            else:
                output = model(input)
            
        acc1, _ = accuracy(output, target, topk=(1, 5))
        if dist:
            acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
            torch.cuda.synchronize()

        top1_m.update(acc1.item(), output.size(0))

    # print(f"Top1 Accuracy on the Stylized-ImageNet: {top1_m.avg:.1f}%")
    return top1_m.avg
