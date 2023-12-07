# -*- coding: utf-8 -*-
"""
This file contains the PyTorch data for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
    "Salinas": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        ],
        "img": "Salinas_corrected.mat",
        "gt": "Salinas_gt.mat",
    },
    "Pavia": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "Houston": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "XA": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "KSC": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
        ],
        "img": "KSC.mat",
        "gt": "KSC_gt.mat",
    },
    "Indian": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        ],
        "img": "Indian_pines_corrected.mat",
        "gt": "Indian_pines_gt.mat",
    },
    "Botswana": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
            "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        ],
        "img": "Botswana.mat",
        "gt": "Botswana_gt.mat",
    },
    "Honghu": {
        "urls": ["pass"],
        "img": "WHU_Hi_HongHu.mat",
        "gt": "WHU_Hi_HongHu_gt.mat"
    }
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


def get_dataset(dataset_name, train_num=10, target_folder="./data/", datasets=DATASETS_CONFIG):
    """Gets the data specified by name and return the related components.
    Args:
        dataset_name: string with the name of the data
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): data configuration dictionary, defaults to prebuilt one
        train_num (Any):
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    dataset = datasets[dataset_name]
    folder = target_folder + datasets[dataset_name].get("folder", dataset_name + "/")

    if dataset_name == "Pavia":
        mat = open_file(folder + "Pavia_{}_split.mat".format(train_num))
        img = mat['input']
        TR, TE = mat['TR'], mat['TE']
        rgb_bands = (55, 41, 12)
        label_values = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]

        ignored_labels = [0]
    elif dataset_name == "Indian":
        # Load the image
        mat = open_file(folder + "Indian_{}_split.mat".format(train_num))
        img = mat['input']
        rgb_bands = (43, 21, 11)
        TR, TE = mat['TR'], mat['TE']
        label_values = [
            "Undefined",
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]

        ignored_labels = [0]
    elif dataset_name == "Honghu":
        mat = open_file(folder + "Honghu_{}_split.mat".format(train_num))
        img = mat['input']
        TR, TE = mat['TR'], mat['TE']

        rgb_bands = (23, 45, 78)
        label_values = ['Unclassified', 'Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage',
                        'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis',
                        'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce', 'Film covered lecttuce',
                        'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree']
        ignored_labels = [0]

    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype="float32")
    # _,h,w=img.shape
    # gt = np.ones(shape=(h,w))
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, TR, TE, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, gt, band, hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["data"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.center_pixel = hyperparams["center_pixel"]
        self.band = band
        supervision = hyperparams["supervision"]
        mask = np.ones_like(gt)
        for l in self.ignored_labels:
            mask[gt == l] = 0
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [(x, y) for x, y in zip(x_pos, y_pos) if p < x < data.shape[0] - p and p < y < data.shape[1] - p]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data_z = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")
        d, h, w = data_z.shape

        data = np.zeros([self.band, h, w])
        data[0:d, :, :] = data_z 
        data = np.asarray(np.copy(data), dtype="float32")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
