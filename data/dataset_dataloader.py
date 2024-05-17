import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

def get_simple_dataloader(root_dir, paths_file, batch_size, mode, num_workers=1, drop_last=False):
    dataset = FaceRecDataset(root_dir, paths_file, mode)
    shuffle = (mode == 'pretrain')
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


def get_dataloader(root_dir, paths_file, batch_size, mode, rank, world_size, sampler=None, rand_channel_aug=0,
                   num_workers=1, modality=None, drop_last=True, return_domain=False):
    dataset = FaceRecDataset(root_dir, paths_file, mode, rand_channel_aug=rand_channel_aug, modality=modality, return_domain=return_domain)
    if sampler is None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(mode == 'pretrain'), drop_last=drop_last)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, sampler=sampler)


def get_dataloader_wtype(root_dir, paths_file, batch_size, mode, rank, world_size, sampler=None, rand_channel_aug=0, num_workers=1, modality=None):
    dataset = FaceRecDataset_wType(root_dir, paths_file, mode, rand_channel_aug=rand_channel_aug, modality=modality)
    if sampler is None:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(mode == 'pretrain'), drop_last=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, sampler=sampler)


class RandomChannelAugmentation(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, img):
        channel_idx = torch.randint(0, 4, (1,))
        if channel_idx == 3:
            return img
        else:
            return img[channel_idx, :, :].repeat(3, 1, 1)


class RandomRedChannelAugmentation(object):
    """
    Convert ndarrays in sample to Tensors.
    items in  : x10, x20, x60, day, year, labels
    items out : x10, x20, x60, day, year, labels
    """
    def __call__(self, img):
        channel_idx = torch.randint(0, 2, (1,))
        if channel_idx == 1:
            return img
        else:
            return img[0, :, :].repeat(3, 1, 1)  # red channel


def TFS(rand_channel_aug=0):
    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if rand_channel_aug == 1:
        transform_list.append(RandomChannelAugmentation())
    if rand_channel_aug == 2:
        transform_list.append(RandomRedChannelAugmentation())
    transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return transforms.Compose(transform_list)


TFS_EVAL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceRecDataset(Dataset):
    def __init__(self, root_dir, paths_file, mode, rand_channel_aug=0, modality=None, return_domain=False):
        self.root_dir = root_dir
        self.paths = pd.read_csv(paths_file, header=None)#.values.tolist()
        if modality is not None:
            self.paths = self.paths[self.paths[0].apply(lambda s: s.split('/')[0]) == modality]
        self.paths = self.paths.values.tolist()
        self.mode = mode
        self.rand_channel_aug = rand_channel_aug
        if self.mode == 'pretrain':
            self.transform = TFS(rand_channel_aug)
        else:
            self.transform = TFS_EVAL
        self.return_domain = return_domain

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, label = self.paths[index]
        img = Image.open(os.path.join(self.root_dir, img_path))
        img = self.transform(img)
        if self.return_domain:
            return img, torch.tensor(label), torch.tensor(img_path.split('/')[0] == 'NIR').to(torch.int64)
        return img, torch.tensor(label)#, img_type



class FaceRecDataset_wType(Dataset):
    def __init__(self, root_dir, paths_file, mode, rand_channel_aug=0, modality=None):
        self.root_dir = root_dir
        self.paths = pd.read_csv(paths_file, header=None)#.values.tolist()
        if modality is not None:
            self.paths = self.paths[self.paths[0].apply(lambda s: s.split('/')[0]) == modality]
        self.paths = self.paths.values.tolist()
        self.mode = mode
        self.rand_channel_aug = rand_channel_aug
        if self.mode == 'pretrain':
            self.transform = TFS(rand_channel_aug)
        else:
            self.transform = TFS_EVAL

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path, label = self.paths[index]
        img = Image.open(os.path.join(self.root_dir, img_path))
        img = self.transform(img)
        img_type = img_path.split('/')[-2]
        try:
            if img_type == 'NIR':
                img_type = torch.tensor(0)
            elif img_type == 'VIS':
                img_type = torch.tensor(1)
        except:
            print(img_type, img_path)
        return img, torch.tensor(label), img_type


if __name__ == "__main__":
    root_dir = '/home/michaila/Data/NIR-VIS/CASIA/crops112/'
    paths_file = '/home/michaila/Data/NIR-VIS/CASIA/crops112/train_paths_fold%d.csv' % 1
    dataset = FaceRecDataset(root_dir, paths_file, 'pretrain')

    img, lab = dataset[0]
