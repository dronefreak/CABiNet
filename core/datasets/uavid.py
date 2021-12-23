#!/usr/bin/python
# -*- encoding: utf-8 -*-


from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
from core.datasets.transform import *


class UAVid(Dataset):
    def __init__(self, params, mode='train'):
        super(UAVid, self).__init__()
        self.mode = mode
        self.config_file = params["dataset_config"]["dataset_config_file"]
        self.ignore_lb = params["dataset_config"]["ignore_idx"]
        self.rootpth = params["dataset_config"]["dataset_path"]
        self.cropsize = tuple(params["dataset_config"]["cropsize"])
        try:
            assert self.mode in ('train', 'val', 'test')
        except AssertionError:
            print(f"[INFO]: Specified {self.mode} mode not in [train, val, test]")
            raise
        try:
            assert os.path.exists(self.rootpth)
        except AssertionError:
            print(f"[INFO]: Specified dataset path {self.rootpth} does not exist!")
            raise

        with open(self.config_file, 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['trainId']: el['color'] for el in labels_info}

        """ Parse Image Directory """
        self.imgs = {}
        imgnames = []
        impth = osp.join(self.rootpth, 'uavid_' + self.mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd, 'Images')
            im_names = os.listdir(fdpth)
            names = [el.replace('.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        """ Parse GT Directory """
        self.labels = {}
        gtnames = []
        gtpth = osp.join(self.rootpth, 'uavid_' + self.mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd, 'Labels')
            lbnames = os.listdir(fdpth)
            #lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        """ Pre-processing and Data Augmentation """
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(self.cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        label = self.convert_labels(label)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for v, k in self.lb_map.items():
            indices_list = np.where(np.all(label == k, axis=-1))
            label[indices_list] = [v, 0, 0]
        return label[:, :, 0]



if __name__ == "__main__":
    from tqdm import tqdm
    with open('../../configs/train_uavid.json', "r") as f:
        params = json.loads(f.read())
    ds = UAVid(params, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(set(uni))

