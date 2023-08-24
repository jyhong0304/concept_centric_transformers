from torch.utils.data import random_split
from .data_utils import split_dataset, ExplanationDataset
import torchvision.datasets as datasets
import os
import numpy as np

class ImageNetDataset(ExplanationDataset):
    def __init__(self, dataset, num_classes=1000):
        self.num_classes = num_classes
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_data(self, idx):
        inputs, label = self.dataset[idx]
        return inputs, label

    def get_explanations(self, idx):
        return np.nan


def get_imagenet(data_dir, train_transform, test_transform, val_size, random_train_subset=False):
    trainset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform)
    test =  datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=test_transform)

    if not random_train_subset:
        val, train = split_dataset(trainset, val_size)
    else:
        val, train = random_split(trainset, [val_size, len(trainset) - val_size])

    return ImageNetDataset(train), ImageNetDataset(val), ImageNetDataset(test)
