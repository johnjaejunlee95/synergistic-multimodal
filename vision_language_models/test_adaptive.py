import argparse
import os
import logging
from contextlib import suppress
from torch.nn.parallel import DistributedDataParallel as DDP

from timm.data import resolve_data_config
from timm.models import safe_model_name, load_checkpoint
from timm.models.layers import convert_splitbn_model

from src.benchmarks import *
from src.models import *
from src.processes.multimodal import *
from src.utils import arguments


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

_logger = logging.getLogger('test')


def main(args):
    
    logging_path = os.path.join(args.output, args.experiment)
    os.makedirs(logging_path, exist_ok=True)
    setup_default_logging(log_path=os.path.join(logging_path, 'log_test.txt'))

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0
    
    
    print("World size", args.world_size)
    random_seed(args.seed, args.rank)
    
    
    if args.language_model == 'bert':
        args.hidden_size = BertModel.from_pretrained("bert-large-uncased", add_pooling_layer=False, cache_dir="/nfs2/jjlee/model_cache").config.hidden_size
    elif args.languiage_model == 'roberta':
        args.hidden_size = RobertaModel.from_pretrained("roberta-large", add_pooling_layer=False, cache_dir="/nfs2/jjlee/model_cache").config.hidden_size
                                            
    
    model = Vision_Transformers(args)

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
    
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
        
    model = model.to('cuda')
        
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV3(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
            model = model_ema.module
    else:
        if args.resume:
            load_checkpoint(model.module, args.resume, use_ema=True)
            

    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
    else:
        amp_autocast = suppress
    
    if args.distributed:
        if args.local_rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)    
        
    in_c_acc, mCE = 0, 0
    in_a_acc, in_r_acc, in_sk_acc, in_v2_acc, in_val_acc, in_style_acc = 0, 0, 0, 0, 0, 0
    
    in_a_acc = evaluate_imagenet_a(model, os.path.join(args.test_dir, 'imagenet-a'), test_batchsize=args.batch_size*2, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    in_r_acc = evaluate_imagenet_r(model, os.path.join(args.test_dir, 'imagenet-r'), test_batchsize=args.batch_size*2, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    in_sk_acc = evaluate_imagenet_sketch(model, os.path.join(args.test_dir, 'imagenet-sketch'), test_batchsize=args.batch_size*2, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    in_v2_acc = evaluate_imagenet_v2(model, os.path.join(args.test_dir, 'imagenetv2'), test_batchsize=args.batch_size*2, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    in_val_acc = evaluate_imagenet_val(model, os.path.join(args.test_dir, 'imagenet-val'), test_batchsize=args.batch_size*2, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    in_style_acc = evaluate_stylized_imagenet(model, os.path.join(args.test_dir, 'imagenet-style'), test_batchsize=args.batch_size, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    # in_c_acc, mCE = evaluate_imagenet_c(model, os.path.join(args.test_dir, 'imagenet-c'),  test_batchsize=args.batch_size, test_transform=None, amp_autocast=amp_autocast, dist=args.distributed)
    if args.local_rank == 0:
        _logger.info(f"\nImageNet:{in_val_acc:.2f}%\nImageNet-V2: {in_v2_acc:.2f}%\nImageNet-R: {in_r_acc:.2f}%\nImageNet-Sketch: {in_sk_acc:.2f}%\nImageNet-A: {in_a_acc:.2f}%\nImageNet-Style: {in_style_acc:.2f}% \nImageNet-C: acc --> {in_c_acc:.2f}% | mCE --> {mCE*100:.2f}")  

    

if __name__ == "__main__":
    
    args, args_text = arguments.parse_args()
    main(args)