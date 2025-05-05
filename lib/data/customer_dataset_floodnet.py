#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np

import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset

floodnet_labels_info = [
    {'id': 0, 'trainId': 0, 'name': 'background'},
    {'id': 1, 'trainId': 1, 'name': 'building-flooded'},
    {'id': 2, 'trainId': 2, 'name': 'building-non-flooded'},
    {'id': 3, 'trainId': 3, 'name': 'road-flooded'},
    {'id': 4, 'trainId': 4, 'name': 'road-non-flooded'},
    {'id': 5, 'trainId': 5, 'name': 'water'},
    {'id': 6, 'trainId': 6, 'name': 'tree'},
    {'id': 7, 'trainId': 7, 'name': 'vehicle'},
    {'id': 8, 'trainId': 8, 'name': 'pool'},
    {'id': 9, 'trainId': 9, 'name': 'grass'},
]

class CustomerDatasetFloodNet(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CustomerDatasetFloodNet, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 10
        self.lb_ignore = 255  # 未标注区域的像素值

        # 标签映射表
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in floodnet_labels_info:  # 根据标签定义进行设置
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.414521723985672, 0.4492538869380951, 0.3439224064350128),  # rgb
            std=(0.2068883776664734, 0.19211505353450775, 0.207492396235466)
        )


