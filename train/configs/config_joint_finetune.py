from easydict import EasyDict as edict


config = edict()
config.nirvis_dataset = "BUAA"
config.fold = 1
config.rgb_dataset = 'MS1M'
config.architecture = 'mobilefacenet'
config.embedding_size = 512
config.lr = 0.0001  # 0.1
config.nirvis_batch_size = 8
config.vis_batch_size = 64
config.num_epoch = 10
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.backbone_checkpoint = ""
config.vis_classifier_checkpoint = ""
config.nirvis_classifier_checkpoint = None
config.update_batchnorm = True
config.update_vis_classifier = False
config.margin = 0.6
config.lambda_vis = 1.0
config.lambda_nirvis = 1.0
config.rgb_random_channel_aug = 2
config.output = '%s_init-MS1MReCA_%s%dof10_m06_lr1e4' % (config.architecture, config.nirvis_dataset, config.fold)


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [10, 15, 20] if m - 1 <= epoch])

config.lr_func = lr_step_func

if config.rgb_dataset == "MS1M":
    config.rgb_root_dir = ''
    config.rgb_paths_file = ''
    config.label2index_path = ''
    config.num_samples = 5162456
    config.vis_num_classes = 93407
    config.weight_decay = 5e-4

if config.nirvis_dataset == "CASIA":
    config.nirvis_root_dir = ''
    config.nirvis_paths_file = '/train_paths_fold%d.csv' % config.fold
    config.nirvis_num_classes = 357

if config.nirvis_dataset == "BUAA":
    config.nirvis_root_dir = ''
    config.nirvis_paths_file = ''
    config.nirvis_num_classes = 110

if config.nirvis_dataset == "LAMP-HQ":
    config.nirvis_root_dir = ''
    config.nirvis_paths_file = '/train_paths_fold%d.csv' % config.fold
    config.nirvis_num_classes = {1: 300, 2: 285, 3: 285, 4: 298, 5: 300, 6: 273, 7: 272, 8: 281, 9: 274, 10: 279}[config.fold]

if config.nirvis_dataset == "OULU-CASIA":
    config.nirvis_root_dir = ''
    config.nirvis_paths_file = ''
    config.nirvis_num_classes = 40