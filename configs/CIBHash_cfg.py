import os

import torch.optim as optim

from .utils import config_dataset


def get_config(start_time):
    config = {
        # "dataset": "mirflickr",
        # "dataset": "cifar10-1",
        # "dataset": "coco",
        "dataset": "nuswide_21",
        # "dataset": "nuswide_10",
        
        "bit_list": [16, 32, 64],

        "info": "CIBHash",
        "backbone": "ViT-B_16",
        "pretrained_dir": "/hy-tmp/checkpoint/ViT-B_16.npz",

        "frozen backbone": False,
        "optimizer": {"type": optim.Adam, 
                      "lr": 0.001,
                      "backbone_lr": 1e-5},
        "epoch": 2,
        "test_map": 2,
        "batch_size": 64, 
        "num_workers": 8,
        "logs_path": "logs",

        "resize_size": 224,
        "crop_size": 224,

        "temperature": 0.3,
        "weight": 0.001,
    }
    config = config_dataset(config)
    config["logs_path"] = os.path.join(config["logs_path"], config['info'], start_time)

    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if 'cifar' in config["dataset"]:
        config["topK"] = 1000
    else: 
        config["topK"] = 5000

    return config