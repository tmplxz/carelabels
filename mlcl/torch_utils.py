import torch
from torchvision.datasets.folder import DatasetFolder
import cv2
from PIL import Image


# https://github.com/pytorch/examples/blob/af111380839d35924e9e36437aeb9757b5a68f96/imagenet/main.py#L411
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VideoFolder(DatasetFolder):

    def __init__(self, root, transform=None, target_transform=None, loader=None):
        super(VideoFolder, self).__init__(
            root, loader, '.mp4', transform=transform, target_transform=target_transform)

        self.vids = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        # cap = VideoCapture(path)
        cap = cv2.VideoCapture(path)

        frames = []

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(Image.fromarray(frame)).unsqueeze(0))

        cap.release()

        return torch.cat(frames, 0), target
