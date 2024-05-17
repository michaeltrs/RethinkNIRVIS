from easydict import EasyDict as edict

config = edict()

# Architecture ---------------------------------------------------------------------------------------------------------
config.architecture = 'mobilefacenet'  # Choose among: 'mobilefacenet', 'lightcnn29v2', 'iresnet18', 'iresnet50', 'iresnet100'
if config.architecture == 'lightcnn29v2':
    emb_size = 256
else:
    emb_size = 512
config.embedding_size = emb_size
config.backbone_checkpoint = None
config.classifier_checkpoint = None
config.output = '%s_MS1M' % config.architecture

# Dataset - MS1M -------------------------------------------------------------------------------------------------------
config.root_dir = ''
config.paths_file = ''
config.label2index_path = ''
config.num_samples = 5162456
config.num_classes = 93407
config.random_channel_aug = 0  # choose among 0: no augmentation, 1: random red channel augmentation

# Training
config.num_epoch = 24
config.weight_decay = 5e-4
config.lr = 0.1
config.batch_size = 2 #64  # batch size per device

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [10, 18, 22] if m - 1 <= epoch])

config.lr_func = lr_step_func
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
