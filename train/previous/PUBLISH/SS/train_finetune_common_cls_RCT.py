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
# from configs.config_iresnet18_commonembed import config as cfg
from losses import FeatureLoss
from dataset_dataloader import get_dataloader
from dataset import MXFaceDataset, DataLoaderX
from sgd import SGD

import json
import types
import shutil


torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

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
        # cosine[~index] += (m_hot/2)
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
    try:
        rand_channel_aug = cfg.random_channel_aug
    except:
        rand_channel_aug = 0

    train_loader = get_dataloader(cfg.rgb_root_dir, cfg.rgb_paths_file, cfg.rgb_batch_size,
                                  rand_channel_aug=cfg.rgb_random_channel_aug, mode='pretrain',
                                  rank=local_rank, world_size=cfg.world_size, num_workers=0)
    train_iter = iter(train_loader)
    train_loader_nirvis = get_dataloader(cfg.nirvis_root_dir, cfg.nirvis_paths_file, cfg.nirvis_batch_size,
                       rand_channel_aug=0, mode='pretrain',
                       rank=local_rank, world_size=cfg.world_size, num_workers=0)
    train_iter_nirvis = iter(train_loader_nirvis)


    # print("num ims: ", len(train_loader))
    # BACKBONE ------------------------------------------------------------------------------------------------------
#     backbone = backbones.iresnet34(False).to(local_rank)
#     backbone = backbones.iresnet100(False).to(local_rank)
#     backbone = backbones.huawei256().to(local_rank)
#     backbone = backbones.IR_conds_18x3([112, 112],4).to(local_rank)
#     backbone = backbones.IR_conds_18x3_dense([112, 112], 4).to(local_rank)
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
    # Margin softmax
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
        # find_unused_parameters=True
    )

    # NIR-VIS CLASSIFIER --------------------------------------------------------------------------------------------
    nirvis_classifier = MarginSoftmaxClassifier(in_features=cfg.embedding_size, out_features=cfg.nirvis_num_classes,
                                             s=64.0, m=0.4).to(local_rank)

    if cfg.nirvis_classifier_checkpoint not in ['', None]:
        print('loading NIR-VIS classifier checkpoint from %s' % cfg.vis_classifier_checkpoint)
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
    # nirvis_classifier.pretrain()

    # Optimizer for backbone and classifer
    # if cfg.update_rgb_classifier:
    optimizer = SGD([
        {'params': backbone.parameters()},
        # {'params': rgb_classifier.parameters()},
        # {'params': nirvis_classifier.parameters()}
    ],
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        rescale=cfg.world_size)
    # else:
    #     optimizer = SGD([
    #         {'params': backbone.parameters()},
    #         # {'params': rgb_classifier.parameters()},
    #         # {'params': nirvis_classifier.parameters()}
    #     ],
    #         lr=cfg.lr,
    #         momentum=0.9,
    #         weight_decay=cfg.weight_decay,
    #         rescale=cfg.world_size)

    # Lr scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=cfg.lr_func)
    n_epochs = cfg.num_epoch
    start_epoch = 0  #

    total_step = int(len(train_loader) / cfg.rgb_batch_size / dist.get_world_size())
    if dist.get_rank() == 0:
        cfg2write = {k: v for k, v in cfg.items() if not isinstance(v, types.FunctionType)}
        if not os.path.exists(cfg.output):
            os.makedirs(cfg.output)
            with open(os.path.join(cfg.output, 'cfg.txt'), 'w') as file:
                file.write(json.dumps(cfg2write))
        else:
            with open(os.path.join(cfg.output, 'cfg2.txt'), 'w') as file:
                file.write(json.dumps(cfg2write))
        shutil.copy(__file__, os.path.join(cfg.output, os.path.basename(__file__)))
        print("Total Step is: %d" % total_step)


    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.output, 'logs/shows'))

    # feature_loss = FeatureLoss()

    # Get average embedding as classifier weights ----------------------------------------------------------------------
    # weights = nirvis_classifier.kernel
    def update_weights_as_mean_embedding():
        weights = torch.zeros((cfg.nirvis_num_classes, cfg.embedding_size)).cuda()
        label_counts = torch.zeros(cfg.nirvis_num_classes).cuda()
        for step, (img, labels) in enumerate(train_loader_nirvis):
            # print("step %d of %d" % (step, len(train_loader)))

            labels = labels.reshape(-1).cpu()
            # with torch.no_grad():
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

        print('Assigned average embeddings to classifier weights')

    update_weights_as_mean_embedding()

    nirvis_classifier.eval()
    rgb_classifier.eval()

    # print(label_counts[:20])
    # print(nirvis_classifier.module.kernel[:5, :5])
    # print(label_counts.unique())
    # print(labels)
    # print(features.shape)
    # print(weights.shape)
    # break

    rgb_cls_losses = AverageMeter()
    nirvis_cls_losses = AverageMeter()
    # vis_cls_losses = AverageMeter()
    # feat_losses = AverageMeter()
    losses = AverageMeter()
    global_step = 0
    train_start = time.time()
    # break_epoch = False

    if not cfg.update_batchnorm:
        backbone.apply(_freeze_norm_stats)

    for epoch in range(start_epoch, n_epochs):

        if (epoch != 0) and (dist.get_rank() == 0):
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'backbone.pth'))

        for step, (nirvis_img, nirvis_labels) in enumerate(train_loader_nirvis):
        # for step, (img, label) in enumerate(train_loader):

            # if (epoch * len(train_loader) + step) % 1000 == 0:
            #     if dist.get_rank() == 0:
            #         torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch * len(train_loader) + step) + 'step_backbone.pth'))
            try:
                img, label = next(train_iter)
            except:
                print('Rebuilding nir-vis small set iterator')
                train_iter = iter(train_loader)
                img, label = next(train_iter)

                # if dist.get_rank() == 0:
                #     torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(nirvis_epoch) + 'backbone.pth'))
                    # torch.save(rgb_classifier.module.state_dict(),
                    #            os.path.join(cfg.output, str(epoch) + 'rgb_classifier.pth'))
                # nirvis_epoch += 1

            Brgb = img.shape[0]
            # Bnirvis = nirvis_img.shape[0]

            # nirvis_labels = nirvis_labels.reshape(-1).to(label.device)
            # print(img.shape, img.dtype, nirvis_img.shape, nirvis_img.dtype)
            # time.sleep(10)

            all_im = torch.cat((img, nirvis_img), dim=0)
            all_feat = backbone(all_im)
            rgb_feat = all_feat[:Brgb]
            nirvis_feat = all_feat[Brgb:]

            # id_features = backbone(img)
            rgb_pred = rgb_classifier(rgb_feat, label)
            rgb_cls_loss = F.cross_entropy(rgb_pred, label.to(rgb_pred.device))

            # nirvis_features = backbone(nirvis_img, return_feats=False)
            nirvis_pred = nirvis_classifier(nirvis_feat, nirvis_labels)
            nirvis_cls_loss = F.cross_entropy(nirvis_pred, nirvis_labels.to(nirvis_feat.device))

            loss = cfg.lambda_nirvis * nirvis_cls_loss + cfg.lambda_rgb *rgb_cls_loss  #
            # print(id_features.device, rgb_pred.device, rgb_cls_loss, nirvis_features.device, nirvis_pred.device, nirvis_cls_loss)
            # time.sleep(10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rgb_cls_losses.update(rgb_cls_loss.item(), 1)
            nirvis_cls_losses.update(nirvis_cls_loss.item(), 1)
            # vis_cls_losses.update(vis_cls_loss.item(), 1)
            # feat_losses.update(feat_loss.item(), 1)
            losses.update(loss.item(), 1)
            # except:
            #     continue

            if cfg.local_rank == 0 and step % 50 == 0:
                time_now = (time.time() - train_start) / 3600
                time_total = time_now / ((global_step + 1) / total_step)
                time_for_end = time_total - time_now
                writer.add_scalar('time_for_end', time_for_end, global_step)
                writer.add_scalar('loss', loss, global_step)
                print("Speed %d samples/sec CLS Loss %.4f NIR-VIS Loss %.4f Loss %.4f   Epoch: %d   Global Step: %d   Required: %1.f hours" %
                      (
                          (cfg.rgb_batch_size * global_step / (time.time() - train_start) * cfg.world_size),
                          rgb_cls_losses.avg,
                          nirvis_cls_losses.avg,
                          losses.avg,
                          epoch,
                          global_step,
                          time_for_end
                      ))
                losses.reset()
                rgb_cls_losses.reset()
                # feat_losses.reset()

            # if dist.get_rank() == 0:
            #     # import os
            #     # if not os.path.exists(cfg.output):
            #     #     os.makedirs(cfg.output)
            #     torch.save(backbone.module.state_dict(), os.path.join(cfg.output, str(epoch) + 'backbone.pth'))
            #     torch.save(rgb_classifier.module.state_dict(),
            #                os.path.join(cfg.output, str(epoch) + 'rgb_classifier.pth'))

            global_step += 1
        scheduler.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--config', type=str, default='iresnet18_init-WF600k_LAMP-HQ1of10_common-cls_RCT', help='config file name')
    # parser.add_argument('--checkpoint', type=str, default='config', help='load backbone from checkpoint')

    args = parser.parse_args()

    # CHECKPOINT -------------------------------------------------------------------------------------------
    # args.checkpoint = '~/Projects/facerec/heterogeneous/pretrain/WF600K__lightcnn29v2/19backbone.pth'
    # args.checkpoint = '~/Projects/facerec/heterogeneous/pretrain/WF600K_cond_ir_18_T30_20_1/19backbone.pth'

    # cfg = get_cfg(args.config)
    cfg = importlib.import_module('configs.train_finetune.commonCLS_RCT.%s' % args.config).config
    # print(cfg)

    main(args.local_rank)
