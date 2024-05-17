import os
import argparse
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from backbones import get_backbone
import importlib
from data.dataset_dataloader import get_dataloader
import json
import types
from utils.train_utils import SGD, AverageMeter, MarginSoftmaxClassifier
torch.backends.cudnn.benchmark = True


def main(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()

    lambda_vis = cfg.lambda_vis

    train_loader = get_dataloader(cfg.rgb_root_dir, cfg.rgb_paths_file, cfg.vis_batch_size,
                                  rand_channel_aug=cfg.rgb_random_channel_aug, mode='pretrain',
                                  rank=local_rank, world_size=cfg.world_size, num_workers=0)

    train_iter_nirvis = iter(
        get_dataloader(cfg.nirvis_root_dir, cfg.nirvis_paths_file, cfg.nirvis_batch_size, mode='pretrain',
                       rank=local_rank, world_size=cfg.world_size, num_workers=0))

    backbone = get_backbone(cfg.architecture)().to(local_rank)

    if cfg.backbone_checkpoint not in ['', None]:
        print('loading backbone checkpoint from %s' % cfg.backbone_checkpoint)
        weight = torch.load(cfg.backbone_checkpoint, map_location='cuda:{}'.format(cfg.local_rank))
        backbone.load_state_dict(weight)

    # Broadcast init parameters
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=cfg.update_batchnorm,
        device_ids=[cfg.local_rank])

    # RGB CLASSIFIER ------------------------------------------------------------------------------------------------
    rgb_classifier = MarginSoftmaxClassifier(in_features=cfg.embedding_size, out_features=cfg.vis_num_classes,
                                         s=64.0, m=0.4).to(local_rank)

    if cfg.vis_classifier_checkpoint not in ['', None]:
        print('loading RGB classifier checkpoint from %s' % cfg.vis_classifier_checkpoint)
        weight = torch.load(cfg.vis_classifier_checkpoint, map_location='cuda:{}'.format(cfg.local_rank))
        rgb_classifier.load_state_dict(weight)

    # Broadcast init parameters
    for ps in rgb_classifier.parameters():
        dist.broadcast(ps, 0)

    # DDP
    rgb_classifier = torch.nn.parallel.DistributedDataParallel(
        module=rgb_classifier,
        broadcast_buffers=False,
        device_ids=[cfg.local_rank],
    )

    # NIR-VIS CLASSIFIER --------------------------------------------------------------------------------------------
    nirvis_classifier = MarginSoftmaxClassifier(in_features=cfg.embedding_size, out_features=cfg.nirvis_num_classes,
                                             s=64.0, m=0.4).to(local_rank)

    if cfg.nirvis_classifier_checkpoint not in ['', None]:
        print('loading classifier checkpoint from %s' % cfg.nirvis_classifier_checkpoint)
        weight = torch.load(cfg.nirvis_classifier_checkpoint, map_location='cuda:{}'.format(cfg.local_rank))

        # Assumes loading a checkpoint trained on ms1m combined with all nirvis datasets in given order
        idx = {'LampHQ1of10': [93407, 93707],
               'CASIA1of10': [93707, 94064],
               'BUAA': [94064, 94174],
               'OuluCASIA': [94174, 94214]}[cfg.nirvis_dataset]
        weight['kernel'] = weight['kernel'][:, idx[0]:idx[1]]
        print(f'dataset: {args.config.split("-")[-1]}, idx: {idx}')

        nirvis_classifier.load_state_dict(weight)

    # Broadcast init parameters
    for ps in nirvis_classifier.parameters():
        dist.broadcast(ps, 0)

    # DDP
    nirvis_classifier = torch.nn.parallel.DistributedDataParallel(
        module=nirvis_classifier,
        broadcast_buffers=cfg.update_batchnorm,
        device_ids=[cfg.local_rank],
    )

    backbone.train()
    rgb_classifier.train()
    nirvis_classifier.train()

    # Optimizer for backbone and classifer
    if cfg.update_vis_classifier:
        optimizer = SGD([
            {'params': backbone.parameters()},
            {'params': rgb_classifier.parameters()},
        ],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            rescale=cfg.world_size)
    else:
        optimizer = SGD([
            {'params': backbone.parameters()},
        ],
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
            rescale=cfg.world_size)

    # Lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=cfg.lr_func)
    n_epochs = cfg.num_epoch
    start_epoch = 0  #

    total_step = int(len(train_loader) * cfg.num_epoch)

    if dist.get_rank() == 0:
        cfg2write = {k: v for k, v in cfg.items() if not isinstance(v, types.FunctionType)}
        if not os.path.exists(cfg.output):
            os.makedirs(cfg.output)
            with open(os.path.join(cfg.output, 'cfg.txt'), 'w') as file:
                file.write(json.dumps(cfg2write))
        else:
            with open(os.path.join(cfg.output, 'cfg2.txt'), 'w') as file:
                file.write(json.dumps(cfg2write))

        print("Total Step is: %d" % total_step)

    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'logs/shows'))

    rgb_cls_losses = AverageMeter()
    nirvis_cls_losses = AverageMeter()
    losses = AverageMeter()
    global_step = 0
    train_start = time.time()

    for epoch in range(start_epoch, n_epochs):

        for step, (img, label) in enumerate(train_loader):

            img = img.to(local_rank)
            label = label.to(local_rank)

            if (epoch * len(train_loader) + step) % 1000 == 0:
                if dist.get_rank() == 0:
                    torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch * len(train_loader) + step) + 'step_backbone.pth'))

            try:
                nirvis_img, nirvis_labels = next(train_iter_nirvis)
            except:
                print('Rebuilding nir-vis small set iterator')
                train_iter_nirvis = iter(
                    get_dataloader(cfg.nirvis_root_dir, cfg.nirvis_paths_file, cfg.nirvis_batch_size, mode='pretrain',
                                   rank=local_rank, world_size=cfg.world_size, num_workers=0))
                nirvis_img, nirvis_labels = next(train_iter_nirvis)

            nirvis_img = nirvis_img.to(label.device)
            nirvis_labels = nirvis_labels.reshape(-1).to(label.device)

            img_data = torch.cat((img, nirvis_img))

            id_features = backbone(img_data)

            rgb_pred = rgb_classifier(id_features[:img.shape[0]], label)
            rgb_cls_loss = F.cross_entropy(rgb_pred, label)

            nirvis_pred = nirvis_classifier(id_features[-nirvis_img.shape[0]:], nirvis_labels)
            nirvis_cls_loss = F.cross_entropy(nirvis_pred, nirvis_labels)

            loss = lambda_vis * rgb_cls_loss + nirvis_cls_loss  # + vis_cls_loss #+ cfg.nirvis_lambda * feat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rgb_cls_losses.update(rgb_cls_loss.item(), 1)
            nirvis_cls_losses.update(nirvis_cls_loss.item(), 1)
            losses.update(loss.item(), 1)

            if cfg.local_rank == 0 and step % 50 == 0:
                time_now = (time.time() - train_start) / 3600
                time_total = time_now / ((global_step + 1) / total_step)
                time_for_end = time_total - time_now
                writer.add_scalar('time_for_end', time_for_end, global_step)
                writer.add_scalar('loss', loss, global_step)
                print("Speed %d samples/sec CLS Loss %.4f NIR-VIS Loss %.4f Loss %.4f   Epoch: %d   Global Step: %d   Required: %1.f hours" %
                      (
                          (cfg.vis_batch_size * global_step / (time.time() - train_start) * cfg.world_size),
                          rgb_cls_losses.avg,
                          nirvis_cls_losses.avg,
                          losses.avg,
                          epoch,
                          global_step,
                          time_for_end
                      ))
                losses.reset()
                rgb_cls_losses.reset()
            global_step += 1
        scheduler.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--config', type=str, default='iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT', help='config file name')

    args = parser.parse_args()

    cfg = importlib.import_module('configs.%s' % args.config).config
    print(cfg)

    main(args.local_rank)
