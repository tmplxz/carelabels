import os

import torch
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np

from .base import DatasetBase
from mlcl.torch_utils import VideoFolder


class ImageNet(DatasetBase):

    def __init__(self, cfg, pin_memory=False, val_batch_size=32, corruption_bs=32, perturbation_bs=2):
        super().__init__(cfg)
        self.rootdir = cfg

        self.pin_memory = pin_memory
        self.val_batch_size = val_batch_size
        self.corruption_bs = corruption_bs
        self.perturbation_bs = perturbation_bs

        val = os.path.join(cfg, 'imagenet', 'val')  # only use validation data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=self.val_batch_size, shuffle=False,
            num_workers=0, pin_memory=self.pin_memory)

    def get_train(self):
        None

    def get_apply(self):
        return self.val_loader

    def info(self):
        info = {}
        info['name'] = 'ImageNet'
        info['cl_name'] = 'ImageNet'
        info['predict_samples'] = 50000
        return info

    # https://github.com/pytorch/examples/blob/af111380839d35924e9e36437aeb9757b5a68f96/imagenet/main.py#L411
    def evaluate(self, prediction):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        trues = 0.0
        total = 0.0
        for y_hat, (_, y) in zip(prediction, self.val_loader):
            trues += np.count_nonzero(y_hat == y)
            total += y.numpy().size
        return {'Accuracy': trues / total}

    def get_corruption(self, corruption, severity):
        path = os.path.join(self.rootdir, 'imagenet-c', corruption, str(severity))
        if not os.path.exists(path):
            return None
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=self.corruption_bs, shuffle=False,
            num_workers=2, pin_memory=self.pin_memory)
        return val_loader

    def get_perturbation(self, perturbation):
        path = os.path.join(self.rootdir, 'imagenet-p', perturbation)
        if not os.path.exists(path):
            return None
        val_loader = torch.utils.data.DataLoader(
            VideoFolder(root=path, transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=self.perturbation_bs, shuffle=False,
            num_workers=1, pin_memory=self.pin_memory)
        return val_loader
