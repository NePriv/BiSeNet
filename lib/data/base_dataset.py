#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

COLOR_TO_CLASS = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (180, 120, 120): 2,
    (160, 150, 20): 3,
    (140, 140, 140): 4,
    (61, 230, 250): 5,
    (0, 82, 255): 6,
    (255, 0, 245): 7,
    (255, 235, 0): 8,
    (4, 250, 7): 9
}


class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_ignore = -100
        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        img, label = self.get_image(impth, lbpth)
        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def get_image(self, impth, lbpth):
        # 加载图像（保持原有逻辑）
        img = cv2.imread(impth)[:, :, ::-1].copy()  # BGR -> RGB

        # 加载标签为 RGB 彩色图
        label_rgb = cv2.imread(lbpth)
        label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)

        # 将 RGB 标签转换为类别 ID
        label = np.zeros((label_rgb.shape[0], label_rgb.shape[1]), dtype=np.int64)
        for rgb, class_id in COLOR_TO_CLASS.items():
            mask = np.all(label_rgb == rgb, axis=-1)
            label[mask] = class_id

        return img, label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
