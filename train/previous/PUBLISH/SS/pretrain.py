import os
import argparse
import importlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from backbones import get_backbone
from dataset_dataloader import get_dataloader
from sgd import SGD
torch.backends.cudnn.benchmark = True
import time
from losses import ArcFace
import json
import types
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(local_rank):

    dist.init_process_group(backend='nccl', init_method='env://')
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    train_loader = get_dataloader(cfg.root_dir, cfg.paths_file, cfg.batch_size, rand_channel_aug=cfg.random_channel_aug,
                                  mode='pretrain', rank=local_rank, world_size=cfg.world_size, num_workers=0)

    backbone = get_backbone(cfg.architecture)().to(local_rank)

    # Broadcast init parameters
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=True,
        device_ids=[cfg.local_rank])

    # Margin softmax
    classifier = ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=64.0, m=0.4).to(local_rank)
    # Broadcast init parameters
    for ps in classifier.parameters():
        dist.broadcast(ps, 0)

    # DDP
    classifier = torch.nn.parallel.DistributedDataParallel(
        module=classifier,
        broadcast_buffers=True,
        device_ids=[cfg.local_rank])

    backbone.train()
    classifier.train()

    # Optimizer for backbone and classifer
    optimizer = SGD([{
        'params': backbone.parameters()
    }, {
        'params': classifier.parameters()
    }],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        rescale=cfg.world_size)

    # Lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=cfg.lr_func)
    n_epochs = cfg.num_epoch
    start_epoch = 0

    total_step = int(len(train_loader) / dist.get_world_size() * cfg.num_epoch)

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
        shutil.copy(__file__, os.path.join(cfg.output, os.path.basename(__file__)))


    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'logs/shows'))

    losses = AverageMeter()
    global_step = 0
    train_start = time.time()
    for epoch in range(start_epoch, n_epochs):
        for step, (img, label) in enumerate(train_loader):

            features = backbone(img)

            pred = classifier(features, label)

            loss = F.cross_entropy(pred, label.to(pred.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), 1)
            if cfg.local_rank == 0 and step % 50 == 0:
                time_now = (time.time() - train_start) / 3600
                time_total = time_now / ((global_step + 1) / total_step)
                time_for_end = time_total - time_now
                writer.add_scalar('time_for_end', time_for_end, global_step)
                writer.add_scalar('loss', loss, global_step)
                print("Speed %d samples/sec   Loss %.4f   Epoch: %d   Global Step: %d   Required: %1.f hours" %
                      (
                          (cfg.batch_size * global_step / (time.time() - train_start) * cfg.world_size),
                          losses.avg,
                          epoch,
                          global_step,
                          time_for_end
                      ))
                losses.reset()
            global_step += 1
        scheduler.step()
        if dist.get_rank() == 0:

            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'backbone.pth'))
            torch.save(classifier.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'classifier.pth'))

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--config', type=str, default='config_iresnet50_MS1M', help='config file name')

    args = parser.parse_args()

    cfg = importlib.import_module('configs.pretrain.%s' % args.config).config

    main(args.local_rank)
