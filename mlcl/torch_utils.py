import torch
from torchvision.datasets.folder import DatasetFolder
import cv2
from PIL import Image


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