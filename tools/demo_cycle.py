import os
import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenet_customer.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./datasets/vaihingen/train',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.414521723985672, 0.4492538869380951, 0.3439224064350128),  # rgb
    std=(0.2068883776664734, 0.19211505353450775, 0.207492396235466)
)

files = [os.path.join(args.img_path, file) for file in os.listdir(args.img_path)]

i: int = 0
for file in files:
    print(file)

    im = cv2.imread(file)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    out = net(im)[0]
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)

    # visualize
    out = out.squeeze().detach().cpu().numpy()
    pred = palette[out]
    cv2.imwrite('./res_cycle/' + str(i) + '_res.jpg', pred)
    i = i + 1
