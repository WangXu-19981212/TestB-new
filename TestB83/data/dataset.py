import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random

identity = lambda x: x


# 用于加载图像数据并应用指定的变换。该类从 JSON 文件中读取图像路径和标签，并在获取数据时对图像和标签进行相应的变换
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])

        # image_path = image_path[:12]+'2'+image_path[13:]
        # image_path = image_path.replace('DATACENTER/4', 'share/test')
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


# 用于加载特定类别的图像数据并应用指定的变换。该类从给定的子元数据中读取图像路径，并在获取数据时对图像和标签进行相应的变换
class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        if len(self.sub_meta) < min_size:
            idxs = [i % len(self.sub_meta) for i in range(min_size)]
            self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


# 用于加载多个类别的图像数据，并为每个类别创建一个子数据集。该类从 JSON 文件中读取图像路径和标签，并为每个类别生成一个 SubDataset 实例，然后使用 DataLoader 加载数据
class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                            pin_memory=False))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


# 用于生成少样本学习任务中的批次采样器的类。主要功能是生成多个“episode”，每个 episode 包含从数据集中随机选择的 n_way 个类别
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

