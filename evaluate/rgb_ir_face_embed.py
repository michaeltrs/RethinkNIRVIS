import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from backbones import get_backbone


def get_dataloader(root_dir, paths_file, batch_size, mode, transform=None, shuffle=False, return_paths=False, num_workers=1, modality=None):
    dataset = FaceImageDataset(root_dir, paths_file, mode, return_paths=return_paths, transform=transform, modality=modality)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class FaceImageDataset(Dataset):
    def __init__(self, root_dir, paths_file, mode, return_paths=False, transform=None, modality=None):
        self.root_dir = root_dir
        self.paths = pd.read_csv(paths_file, header=None)#.values.tolist()
        if modality is not None:
            self.paths = self.paths[self.paths[0].apply(lambda s: s.split('/')[0]) == modality]
        self.paths = self.paths.values.tolist()
        self.mode = mode
        self.return_paths = return_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index][0]
        img = cv2.imread(os.path.join(self.root_dir, img_path))
        try:
            sample = {'img': torch.tensor(img).permute(2, 0, 1).flip(dims=(0,)).to(torch.float32).div_(255).sub_(0.5).div_(0.5)}
        except:
            print(os.path.join(self.root_dir, img_path))

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, img_path

        return sample


def get_face_embeddings(cfg, root_dir, paths_file, gpu_ids=[0]):

    checkpoint = cfg['checkpoint']
    bs = 100

    device = torch.device("cuda:%d" % gpu_ids[0])

    model = get_backbone(cfg['architecture'])().to(device)

    params = torch.load(checkpoint)
    model.load_state_dict(params)

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.eval()

    exampleloader_vis = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                 batch_size=bs, mode='eval', transform=None, shuffle=False,
                                 return_paths=True, num_workers=4, modality='VIS')
    exampleloader_nir = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                       batch_size=bs, mode='eval', transform=None, shuffle=False,
                                       return_paths=True, num_workers=4, modality='NIR')
    NAMES_VIS = []
    EMBS_VIS = []
    for iteration, sample in enumerate(exampleloader_vis):

        sample, path = sample
        img = sample['img'].to(device)

        emb = F.normalize(model(img), dim=1).cpu().detach().numpy()

        NAMES_VIS.append(np.array(path))
        EMBS_VIS.append(emb)

    EMBS_VIS = np.concatenate(EMBS_VIS)
    NAMES_VIS = np.concatenate(NAMES_VIS)
    EMBS_VIS = pd.DataFrame(data=EMBS_VIS, index=NAMES_VIS)

    NAMES_NIR = []
    EMBS_NIR = []
    for iteration, sample in enumerate(exampleloader_nir):

        sample, path = sample
        img = sample['img'].to(device)

        emb = F.normalize(model(img), dim=1).cpu().detach().numpy()

        NAMES_NIR.append(np.array(path))
        EMBS_NIR.append(emb)

    EMBS_NIR = np.concatenate(EMBS_NIR)
    NAMES_NIR = np.concatenate(NAMES_NIR)
    EMBS_NIR = pd.DataFrame(data=EMBS_NIR, index=NAMES_NIR)

    return pd.concat((EMBS_VIS, EMBS_NIR)).reset_index()



def get_face_embeddings_singlemode(architecture, checkpoint, root_dir, paths_file, gpu_ids=[0]):

    bs = 100

    device = torch.device("cuda:%d" % gpu_ids[0])

    if 'efficientnet' in architecture:
        model = get_backbone(architecture).to(device)
    else:
        model = get_backbone(architecture)().to(device)

    print(checkpoint)
    params = torch.load(checkpoint)
    model.load_state_dict(params)

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.eval()

    exampleloader = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                 batch_size=bs, mode='eval', transform=None, shuffle=False,
                                 return_paths=True, num_workers=4)
    NAMES = []
    EMBS = []
    for iteration, sample in enumerate(exampleloader):

        sample, path = sample
        img = sample['img'].to(device)

        emb = model(img)
        emb = F.normalize(emb, dim=1).cpu().detach().numpy()

        NAMES.append(np.array(path))
        EMBS.append(emb)

    EMBS = np.concatenate(EMBS)
    NAMES = np.concatenate(NAMES)
    EMBS = pd.DataFrame(data=EMBS, index=NAMES)

    return EMBS



def get_face_embeddings_doublemode(architectures, checkpoints, root_dir, paths_file, gpu_ids=[0]):

    arch_vis, arch_nir = architectures
    checkpoint_vis, checkpoint_nir = checkpoints

    bs = 100

    device = torch.device("cuda:%d" % gpu_ids[0])

    model_vis = get_backbone(arch_vis)().to(device)
    model_nir = get_backbone(arch_nir)().to(device)

    print('loading nir models from ', checkpoint_nir)
    params_nir = torch.load(checkpoint_nir)
    model_nir.load_state_dict(params_nir)

    if len(gpu_ids) > 1:
        model_nir = nn.DataParallel(model_nir, device_ids=gpu_ids)

    print('loading vis models from ', checkpoint_vis)
    params_vis = torch.load(checkpoint_vis)
    model_vis.load_state_dict(params_vis)

    if len(gpu_ids) > 1:
        model_vis = nn.DataParallel(model_vis, device_ids=gpu_ids)

    model_vis.eval()
    model_nir.eval()

    exampleloader_vis = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                       batch_size=bs, mode='eval', transform=None, shuffle=False,
                                       return_paths=True, num_workers=4, modality='VIS')
    exampleloader_nir = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                       batch_size=bs, mode='eval', transform=None, shuffle=False,
                                       return_paths=True, num_workers=4, modality='NIR')
    NAMES_VIS = []
    EMBS_VIS = []
    for iteration, sample in enumerate(exampleloader_vis):

        sample, path = sample
        img = sample['img'].to(device)

        emb = F.normalize(model_vis(img), dim=1).cpu().detach().numpy()

        NAMES_VIS.append(np.array(path))
        EMBS_VIS.append(emb)

    EMBS_VIS = np.concatenate(EMBS_VIS)
    NAMES_VIS = np.concatenate(NAMES_VIS)
    EMBS_VIS = pd.DataFrame(data=EMBS_VIS, index=NAMES_VIS)

    NAMES_NIR = []
    EMBS_NIR = []
    for iteration, sample in enumerate(exampleloader_nir):

        sample, path = sample
        img = sample['img'].to(device)

        emb = F.normalize(model_nir(img), dim=1).cpu().detach().numpy()

        NAMES_NIR.append(np.array(path))
        EMBS_NIR.append(emb)

    EMBS_NIR = np.concatenate(EMBS_NIR)
    NAMES_NIR = np.concatenate(NAMES_NIR)
    EMBS_NIR = pd.DataFrame(data=EMBS_NIR, index=NAMES_NIR)

    return pd.concat((EMBS_VIS, EMBS_NIR))

class FeatTransform(torch.nn.Module):
    def __init__(self, n_feature):
        super(FeatTransform, self).__init__()
        self.W = torch.nn.Linear(n_feature, n_feature)  # hidden layer
        self.W.weight.data.copy_(torch.eye(n_feature))

    def forward(self, x):
        return self.W(x) + x


def get_face_embeddings_singlemode2(architecture, checkpoint, root_dir, paths_file, gpu_ids=[0]):

    bs = 100

    device = torch.device("cuda:%d" % gpu_ids[0])

    # ------------------------------------------------------------------------------------------------------------------
    model = get_backbone(architecture)().to(device)

    params = torch.load(checkpoint[0])
    model.load_state_dict(params)

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    feat_transf = FeatTransform(512).to(device)

    params = torch.load(checkpoint[1])
    feat_transf.load_state_dict(params)

    if len(gpu_ids) > 1:
        feat_transf = nn.DataParallel(feat_transf, device_ids=gpu_ids)

    feat_transf.eval()

    exampleloader = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                 batch_size=bs, mode='eval', transform=None, shuffle=False,
                                 return_paths=True, num_workers=4)
    NAMES = []
    EMBS = []
    for iteration, sample in enumerate(exampleloader):

        sample, path = sample
        img = sample['img'].to(device)

        emb = model(img)
        emb = feat_transf(emb)
        emb = F.normalize(emb, dim=1).cpu().detach().numpy()

        NAMES.append(np.array(path))
        EMBS.append(emb)

    EMBS = np.concatenate(EMBS)
    NAMES = np.concatenate(NAMES)
    EMBS = pd.DataFrame(data=EMBS, index=NAMES)

    return EMBS



def get_face_embeddings_singlemode_wUNetInput(architecture, input_architecture, checkpoint, input_checkpoint,
                                              root_dir, paths_file, gpu_ids=[0]):

    bs = 100
    device = torch.device("cuda:%d" % gpu_ids[0])

    model = get_backbone(architecture)().to(device)
    input_model = get_backbone(input_architecture)().to(device)

    print(checkpoint)
    params = torch.load(checkpoint)
    model.load_state_dict(params)

    params = torch.load(input_checkpoint)
    input_model.load_state_dict(params)

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.eval()

    if len(gpu_ids) > 1:
        input_model = nn.DataParallel(input_model, device_ids=gpu_ids)

    input_model.eval()

    exampleloader = get_dataloader(root_dir=root_dir, paths_file=paths_file,
                                 batch_size=bs, mode='eval', transform=None, shuffle=False,
                                 return_paths=True, num_workers=4)
    NAMES = []
    EMBS = []
    for iteration, sample in enumerate(exampleloader):

        sample, path = sample
        img = sample['img'].to(device)

        emb = model(input_model(img) + img)
        emb = F.normalize(emb, dim=1).cpu().detach().numpy()

        NAMES.append(np.array(path))
        EMBS.append(emb)

    EMBS = np.concatenate(EMBS)
    NAMES = np.concatenate(NAMES)
    EMBS = pd.DataFrame(data=EMBS, index=NAMES)

    return EMBS
