#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np

import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset

vaihingen_labels_info = [
    {'id': 0, 'trainId': 0, 'name': 'Impervious surfaces'},
    {'id': 1, 'trainId': 1, 'name': 'Building'},
    {'id': 2, 'trainId': 2, 'name': 'Low vegetation'},
    {'id': 3, 'trainId': 3, 'name': 'Tree'},
    {'id': 4, 'trainId': 4, 'name': 'Car'},
    {'id': 5, 'trainId': 5, 'name': 'Clutter/background'},
]

class CustomerDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CustomerDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        # Vaihingen 数据集类别数量和忽略标签
        self.n_cats = 6  # Vaihingen 有 6 个类别
        self.lb_ignore = 255  # Vaihingen 未标注区域的像素值

        # 标签映射表
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in vaihingen_labels_info:  # 根据 Vaihingen 的标签定义进行设置
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(123.675, 116.28, 103.53),  # rgb
            std=(58.395, 57.12, 57.375)
        )


