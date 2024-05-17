import os
import argparse
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from backbones import get_backbone
import importlib
from dataset_dataloader import get_dataloader
# from dataset import MXFaceDataset, DataLoaderX
from sgd import SGD
import json
import types

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
        cosine = torch.mm(embedding_norm, kernel_norm)
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


def _freeze_norm_stats(net):
    try:
        for m in net.modules():

            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False
                m.eval()
    except ValueError:
        print("error")
        return


def main(local_rank):
    dist.init_process_group(backend='nccl', init_method='env://')
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    train_loader = get_dataloader(cfg.nirvis_root_dir, cfg.nirvis_paths_file, cfg.batch_size, mode='pretrain',
                       rank=local_rank, world_size=cfg.world_size, num_workers=0)

    # print("num ims: ", len(train_loader))
    # BACKBONE ------------------------------------------------------------------------------------------------------
    backbone = get_backbone(cfg.architecture)().to(local_rank)

    if cfg.vis_backbone_checkpoint not in ['', None]:
        print('loading backbone checkpoint from %s' % cfg.vis_backbone_checkpoint)
        backbone_weight = torch.load(cfg.vis_backbone_checkpoint, map_location='cuda:{}'.format(cfg.local_rank))
        backbone.load_state_dict(backbone_weight)

    # Broadcast init parameters
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    # DDP
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone,
        broadcast_buffers=cfg.update_batchnorm,
        device_ids=[cfg.local_rank])

    # NIR-VIS CLASSIFIER --------------------------------------------------------------------------------------------
    nirvis_classifier = MarginSoftmaxClassifier(in_features=cfg.embedding_size, out_features=cfg.nirvis_num_classes,
                                             s=64.0, m=0.4).to(local_rank)

    if cfg.nirvis_classifier_checkpoint not in ['', None]:
        print('loading classifier checkpoint from %s' % cfg.vis_classifier_checkpoint)
        weight = torch.load(cfg.nirvis_classifier_checkpoint, map_location='cuda:{}'.format(cfg.local_rank))
        nirvis_classifier.load_state_dict(weight)

    # Broadcast init parameters
    for ps in nirvis_classifier.parameters():
        dist.broadcast(ps, 0)

    # DDP
    nirvis_classifier = torch.nn.parallel.DistributedDataParallel(
        module=nirvis_classifier,
        broadcast_buffers=False,
        device_ids=[cfg.local_rank],
        # find_unused_parameters=True
    )

    backbone.train()
    # rgb_classifier.pretrain()
    nirvis_classifier.train()

    # Optimizer for backbone and classifer
    optimizer = SGD([
        {'params': backbone.parameters()},
        # {'params': nirvis_classifier.parameters()}
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

    total_step = int(len(train_loader) / cfg.batch_size / dist.get_world_size() * cfg.num_epoch)

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

    # feature_loss = FeatureLoss()

    # UPDATE BATCH-NORM PARAMETERS -------------------------------------------------------------------------------------
    if not cfg.update_batchnorm:
        backbone.apply(_freeze_norm_stats)


    # Get average embedding as classifier weights ----------------------------------------------------------------------
    # weights = nirvis_classifier.kernel
    weights = torch.zeros((cfg.nirvis_num_classes, cfg.embedding_size)).cuda()
    label_counts = torch.zeros(cfg.nirvis_num_classes).cuda()
    for step, (img, labels) in enumerate(train_loader):
        # print("step %d of %d" % (step, len(train_loader)))

        labels = labels.reshape(-1).cpu()
        features = backbone(img, return_feats=False)
        for i, l in enumerate(labels):
            weights[l] += features[i].detach()
            label_counts[l] += 1

    torch.distributed.all_reduce(weights)  # , dst=0)
    torch.distributed.all_reduce(label_counts)  # , dst=0)

    weights = weights / label_counts.unsqueeze(-1)

    # print(weights.shape, label_counts.shape)
    with torch.no_grad():
        nirvis_classifier.module.kernel.copy_(weights.permute(1, 0))
    # print(label_counts[:20])
    # print(nirvis_classifier.module.kernel[:5, :5])
        # print(label_counts.unique())
        # print(labels)
        # print(features.shape)
        # print(weights.shape)
        # break
    print('Assigned average embeddings to classifier weights')
    # ------------------------------------------------------------------------------------------------------------------

    # nir_cls_losses = AverageMeter()
    # vis_cls_losses = AverageMeter()
    losses = AverageMeter()
    global_step = 0
    train_start = time.time()
    # break_epoch = False

    for epoch in range(start_epoch, n_epochs):
        # if break_epoch:
        #     break
        # train_sampler.set_epoch(epoch)
        for step, (img, labels) in enumerate(train_loader):

            # if (epoch * len(train_loader) + step) % 1000 == 0:
            #     if dist.get_rank() == 0:
            #         torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch * len(train_loader) + step) + 'step_backbone.pth'))

            labels = labels.reshape(-1).cuda()  # .to(nir_img.device)

            # nir_features = backbone(nir_img, return_feats=False)
            # vis_features = backbone(vis_img, return_feats=False)
            #
            # vis_pred = nirvis_classifier(vis_features, nirvis_labels)
            # nir_pred = nirvis_classifier(nir_features, nirvis_labels)
            #
            # # print(vis_pred.device, nir_pred.device, nirvis_labels.device)
            # vis_cls_loss = F.cross_entropy(vis_pred, nirvis_labels)
            # nir_cls_loss = F.cross_entropy(nir_pred, nirvis_labels)
            #
            # # feat_loss = feature_loss(nir_features, vis_features)
            #
            # loss = 0.5 * (nir_cls_loss + vis_cls_loss)

            features = backbone(img, return_feats=False)
            pred = nirvis_classifier(features, labels)

            reg = torch.tensor(0.).cuda()
            for i, model_param in enumerate(backbone.module.state_dict().keys()):
                reg += torch.sum(torch.pow(backbone.module.state_dict()[model_param] - backbone_weight[model_param], 2))

            loss = F.cross_entropy(pred, labels) + 0.1 * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # nir_cls_losses.update(nir_cls_loss.item(), 1)
            # vis_cls_losses.update(vis_cls_loss.item(), 1)
            losses.update(loss.item(), 1)

            if cfg.local_rank == 0 and step % 50 == 0:
                time_now = (time.time() - train_start) / 3600
                time_total = time_now / ((global_step + 1) / total_step)
                time_for_end = time_total - time_now
                writer.add_scalar('time_for_end', time_for_end, global_step)
                writer.add_scalar('loss', loss, global_step)
                print("Speed %d samples/sec Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" %
                      (
                          (cfg.batch_size * global_step / (time.time() - train_start) * cfg.world_size),
                          # vis_cls_losses.avg,
                          # nir_cls_losses.avg,
                          losses.avg,
                          epoch,
                          global_step,
                          time_for_end
                      ))
                losses.reset()
            global_step += 1
        if dist.get_rank() == 0:
            torch.save(backbone.module.state_dict(),
                       os.path.join(cfg.output, str(epoch * len(train_loader) + step) + 'step_backbone.pth'))

        scheduler.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--config', type=str, default='iresnet18_init-WF600k_LAMP-HQ1of10_RCT_withreg', help='config file name')
    # parser.add_argument('--checkpoint', type=str, default='config', help='load backbone from checkpoint')

    args = parser.parse_args()

    # CHECKPOINT -------------------------------------------------------------------------------------------
    # args.checkpoint = '~/Projects/facerec/heterogeneous/pretrain/WF600K__lightcnn29v2/19backbone.pth'
    # args.checkpoint = '~/Projects/facerec/heterogeneous/pretrain/WF600K_cond_ir_18_T30_20_1/19backbone.pth'

    # cfg = get_cfg(args.config)
    cfg = importlib.import_module('configs.train_finetune.RCT.%s' % args.config).config
    print(args.config, cfg.output)

    main(args.local_rank)
