import time
from collections import OrderedDict

import torch
import numpy as np 
from timm.utils import *
from src.utils.utils import *



def train_one_epoch(args, epoch, model, tokenizer, language_model, loader, optimizer, loss_fn, lr_scheduler=None, saver=None, loss_scaler=None, amp_autocast=None, model_ema=None, mixup_fn=None, logger=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    latent_loss_m = AverageMeter()
    cls_loss_m = AverageMeter()
    
    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
        
    for batch_idx, (inputs, targets) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        random_idx = torch.randperm(1000)[:len(inputs)].tolist() # ImageNet class num
        class_name = ["This is about class {}.".format(texts) for texts in random_idx]
                        
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        
        with amp_autocast():
            text_tokenized = tokenizer(class_name, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt").to("cuda")
            text_embeddings = language_model(**text_tokenized)
            text_embeddings, _ = text_global_pool(text_embeddings["last_hidden_state"], text_tokenized['input_ids'], 'first')
        
            features, output = model(inputs, is_test=False)
            
            latent_loss = F.mse_loss(features, text_embeddings) * args.lam
            cls_loss = loss_fn(output, targets) * (1- args.lam) 
            total_loss = cls_loss + latent_loss 
        
        if not args.distributed:
            losses_m.update(total_loss.item(), inputs.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                total_loss, 
                optimizer,
                clip_grad=args.clip_grad, 
                clip_mode=args.clip_mode,
                parameters = model.parameters(),
                create_graph=second_order)
        else:
            total_loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(model.parameters())
            optimizer.step()
            
        
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or (batch_idx+1) % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(total_loss.data, args.world_size)
                reduced_latent_loss = reduce_tensor(latent_loss.data, args.world_size)
                reduced_cls_loss = reduce_tensor(cls_loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inputs.size(0))
                latent_loss_m.update(reduced_latent_loss.item(), inputs.size(0))
                cls_loss_m.update(reduced_cls_loss.item(), inputs.size(0))

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Total Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Latent Loss: {latent_loss.val:#.4g} ({latent_loss.avg:#.3g})  '
                    'CLS Loss: {cls_loss.val:#.4g} ({cls_loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch+1,
                        batch_idx+1, len(loader),
                        100. * (batch_idx+1) / (last_idx+1),
                        loss=losses_m,
                        latent_loss = latent_loss_m,
                        cls_loss = cls_loss_m,
                        batch_time=batch_time_m,
                        rate=inputs.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if saver is not None and args.recovery_interval and (last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', np.round(losses_m.avg,4)), ('latent_loss', np.round(latent_loss_m.avg, 4)), ('cls_loss', np.round(cls_loss_m.avg, 4))])


@torch.no_grad()
def validate(args, model, tokenizer, language_model, loader, loss_fn, log_suffix='', logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    latent_loss_m = AverageMeter()
    cls_loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, (inputs, targets) in enumerate(loader):
        last_batch = batch_idx == last_idx
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            
        
        random_idx = torch.randperm(1000)[:len(inputs)].tolist()
        class_name = ["This is about class {}.".format(texts) for texts in random_idx]
        
        text_tokenized = tokenizer(class_name, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt").to("cuda")
        text_embeddings = language_model(**text_tokenized)
        text_embeddings, _ = text_global_pool(text_embeddings["last_hidden_state"], text_tokenized['input_ids'], 'first')
        
        features, output = model(inputs, is_test=False)
                        
        if isinstance(output, (tuple, list)):
            output = output[0]

        # augmentation reduction
        reduce_factor = args.tta
        if reduce_factor > 1:
            output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            targets = targets[0:targets.size(0):reduce_factor]

        latent_loss = F.mse_loss(features, text_embeddings) * args.lam
        cls_loss = loss_fn(output, targets) * (1- args.lam)
        loss = cls_loss + latent_loss 
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            reduced_latent_loss = reduce_tensor(latent_loss, args.world_size)
            reduced_ls_loss = reduce_tensor(cls_loss, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data
            reduced_latent_loss = latent_loss
            reduced_ls_loss = cls_loss

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), inputs.size(0))
        latent_loss_m.update(reduced_latent_loss.item(), inputs.size(0))
        cls_loss_m.update(reduced_ls_loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()
        if args.local_rank == 0 and (last_batch or (batch_idx+1) % args.log_interval == 0):
            log_name = 'Test' + log_suffix
            logger.info(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Latent Loss: {latent_loss.val:#.4g} ({latent_loss.avg:#.3g})  '
                'CLS Loss: {cls_loss.val:#.4g} ({cls_loss.avg:#.3g})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                    log_name, 
                    batch_idx+1, 
                    last_idx+1, 
                    batch_time=batch_time_m,
                    loss=losses_m, 
                    latent_loss = latent_loss_m,
                    cls_loss = cls_loss_m, 
                    top1=top1_m, 
                    top5=top5_m))


    metrics = OrderedDict([('loss', np.round(losses_m.avg,4)), ('latent_loss', np.round(latent_loss_m.avg, 4)), ('cls_loss', np.round(cls_loss_m.avg, 4)), ('top1', np.round(top1_m.avg,4)), ('top5', np.round(top5_m.avg,4))])
    return metrics