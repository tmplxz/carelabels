import torch
from torchvision import models as torch_models

from mlcl.implementations import BaseImplementation
from mlcl.datasets import ImageNet


MODELNAME_MAP = {
    'AlexNet': 'alexnet',
    'ResNet-18': 'resnet18',
    'MobileNetV3_Large': 'mobilenet_v3_large',
    'VGG11': 'vgg11'
}


class PretrainedImageNetDNN(BaseImplementation):

    def __init__(self):
        self.config = {}
        self.ds_class = ImageNet
        self.cl_file = 'mlcl/carelabels/cl_dnn.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # METHOD TO OVERRIDE:

    def prepare(self, args):
        self.modelname = args['model']
        self.gpu = args['gpu']
        self.model = torch_models.__dict__[MODELNAME_MAP[self.modelname]](pretrained=True)

    def get_info(self):
        cfg = {
            'config_id': '_'.join([self.modelname, 'gpu' if self.gpu else 'cpu']),
            'uses_gpu': self.gpu
        }
        return cfg

    def get_meta_info(self):
        info = {
            'platform': 'CPU Intel Core i7-10610U',
            'software': 'Python, PyTorch',
            'Metric': 'Top-1 Accuracy'
        }
        return info
    
    def train(self, data, ds_info=None):
        # no training, uses pretrained models
        pass 

    def apply(self, data, ds_info=None):
        self.model.eval()
        self.model.to(self.device)
        preds = []
        for x, y in data:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(x).max(1)[1]
            preds.append(y_hat)
        return preds
