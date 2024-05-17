from easydict import EasyDict as edict

config = edict()
config.nirvis_dataset = "BUAA"  # "CASIA"  #'NIRVIS'  #  has to be a NIR-VIS dataset
config.fold = 1
config.architecture = 'iresnet100'
config.embedding_size = 512
config.lr = 0.0001  # 0.1
config.batch_size = 8
config.num_epoch = 40
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.vis_backbone_checkpoint = "/home/michaila/Projects/facerec/heterogeneous/pretrain/iresnet100_checkpoints/iresnet100_MS1M_ReCA/23backbone.pth"
config.nirvis_classifier_checkpoint = None
config.update_batchnorm = True
# config.output = '%s_init-MS1M_finetune%s%dof10_RCT_lr1e5_MS1Mdistill_LampHQfinetune_2' % (config.architecture, config.nirvis_dataset, config.fold)
config.output = '%s_init-MS1MReCA_RCT_%s_lr1e4' % (config.architecture, config.nirvis_dataset)


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
    config.nirvis_paths_file = '/home/michaila/Data/NIR-VIS/BUAAVISNIR/crops112_3/train_paths_nirvis.csv'
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