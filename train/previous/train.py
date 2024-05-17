import argparse
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import backbones
# from config import config as cfg
from configs.config_iresnet18 import config as cfg
from dataset import MXFaceDataset, DataLoaderX
from sgd import SGD

torch.backends.cudnn.benchmark = True


class MarginSoftmaxClassifier(nn.Module):
    def __init__(self, in_features,out_features,s=64.0, m=0.40):
        super(MarginSoftmaxClassifier, self).__init__()
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)
        self.s = s
        self.m = m

    def forward(self, embedding, label):
        embedding_norm = F.normalize(embedding,dim=1)
        kernel_norm = F.normalize(self.kernel,dim=0)
        cosine = torch.mm(embedding_norm,kernel_norm)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0],
                            cosine.size()[1],
                            device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


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
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(local_rank=local_rank,
                               dataset=trainset,
                               batch_size=cfg.batch_size,
                               sampler=train_sampler,
                               num_workers=0,
                               pin_memory=True,
                               drop_last=False)

#     backbone = backbones.iresnet34(False).to(local_rank)
#     backbone = backbones.iresnet100(False).to(local_rank)
#     backbone = backbones.huawei256().to(local_rank)
    backbone = backbones.IR_conds_18x3([112,112],4).to(local_rank)
    # Broadcast init parameters
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=False,
        device_ids=[cfg.local_rank])

    # Margin softmax
    classifier = MarginSoftmaxClassifier(in_features=cfg.embedding_size,out_features=cfg.num_classes,s=64.0, m=0.4).to(local_rank)
    # Broadcast init parameters
    for ps in classifier.parameters():
        dist.broadcast(ps, 0)

    # DDP
    classifier = torch.nn.parallel.DistributedDataParallel(
        module=classifier,
        broadcast_buffers=False,
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

    if local_rank == 0:
        writer = SummaryWriter(log_dir='logs/shows')

    #
    total_step = int(len(trainset) / cfg.batch_size / dist.get_world_size() * cfg.num_epoch)
    if dist.get_rank() == 0:
        print("Total Step is: %d" % total_step)

    losses = AverageMeter()
    global_step = 0
    train_start = time.time()
    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            if epoch==0:
                dynamic_T = 30
            elif epoch==1:
                dynamic_T = 10
            else:
                dynamic_T = 1
            
            features = backbone(img,dynamic_T)
            pred = classifier(features,label)
            loss = F.cross_entropy(pred,label)
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
            import os
            if not os.path.exists(cfg.output):
                os.makedirs(cfg.output)
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'backbone.pth'))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()
    main(args.local_rank)
