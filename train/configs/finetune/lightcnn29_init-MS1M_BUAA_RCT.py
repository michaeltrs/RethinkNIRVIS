from easydict import EasyDict as edict

config = edict()
config.nirvis_dataset = "BUAA"  # "CASIA"  #'NIRVIS'  #  has to be a NIR-VIS dataset
config.fold = 1
config.architecture = 'lightcnn29v2'
config.embedding_size = 256
config.lr = 0.0001  # 0.1
config.batch_size = 8
config.num_epoch = 40
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.vis_backbone_checkpoint = "/home/michaila/Projects/facerec/heterogeneous/pretrain/lightcnn29v2_MS1M/23backbone.pth"
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/mobilefacenet_checkpoints/mobilefacenet_MS1M_ReCA/23backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/mobilefacenet_MS1M_ReCA/23backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/T_iresnet50_S_mobilefacenet_CLS_MS1MLampHQ_SimKD_MS1M_cls/22backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/T_iresnet50_S_mobilefacenet_CLS_MS1M_SimKD_MS1M_cls_ReCA_fromaligned_lr1e4/1backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/T_iresnet50_S_mobilefacenet_CLS_MS1MLampHQ_598step_SimKD_MS1M_cls_lr1e3_ReCA/0backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/T_iresnet50_S_mobilefacenet_CLS_MS1MLampHQ_200step_SimKD_MS1M_cls_lr1e2_ReCA/1backbone.pth'
    # '/home/michaila/Projects/facerec/heterogeneous/pretrain/T_iresnet50_S_mobilefacenet_CLS_MS1M_SimKD_MS1M/23backbone.pth'
config.nirvis_classifier_checkpoint = None
config.update_batchnorm = True
config.margin = 0.8
config.output = '%s_init-MS1MReCA_finetune%s_RCT_m08_lr1e4' % (config.architecture, config.nirvis_dataset)


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [10, 15, 20] if m - 1 <= epoch])

config.lr_func = lr_step_func

if config.nirvis_dataset == "CASIA":
    config.nirvis_root_dir = '/home/michaila/Data/NIR-VIS/CASIA/crops112/'
    config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/CASIA/crops112/train_paths_fold%d.csv' % config.fold
    config.nirvis_num_classes = 357

if config.nirvis_dataset == "BUAA":
    config.nirvis_root_dir = '/home/michaila/Data/NIR-VIS/BUAAVISNIR/crops112_3/'
    config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/BUAAVISNIR/crops112_3/train_nirvis_paths.csv'
    config.nirvis_num_classes = 110

if config.nirvis_dataset == "LAMP-HQ":
    config.nirvis_root_dir = '/home/michaila/Data/NIR-VIS/LAMP-HQ/crops112/'
    # config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/LAMP-HQ/crops112/train_paths_nirvis_pairs_fold%d.csv' % config.fold
    config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/LAMP-HQ/crops112/train_paths_fold%d.csv' % config.fold
    config.nirvis_num_classes = {1: 300, 2: 285, 3: 285, 4: 298, 5: 300, 6: 273, 7: 272, 8: 281, 9: 274, 10: 279}[config.fold]

if config.nirvis_dataset == "OULU-CASIA":
    config.nirvis_root_dir = '/home/michaila/Data/NIR-VIS/Oulu_CASIA_NIR_VIS/crops112_3/'
    config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/Oulu_CASIA_NIR_VIS/crops112_3/train_paths_nirvis.csv'
    config.nirvis_num_classes = 40