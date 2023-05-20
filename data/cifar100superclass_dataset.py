import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR100
from .data_utils import ExplanationDataset, split_dataset


class CIFAR100SuperclassDataset(ExplanationDataset):
    def __init__(self, dataset, num_superclasses=20, num_classes=100):
        self.num_superclasses = num_superclasses
        self.num_classes = num_classes
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_data(self, idx):
        inputs, label = self.dataset[idx]

        return inputs, self._sparse2coarse(label)

    def get_explanations(self, idx):
        _, label = self.dataset[idx]
        explanation_onehot = F.one_hot(torch.tensor(label).long(), num_classes=self.num_classes).float()

        return explanation_onehot

    def _sparse2coarse(self, targets):
        """
        Convert Pytorch CIFAR100 classes to super-classes.
        Implement this with the reference: https://github.com/ryanchankh/cifar100coarse
        """
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

        return coarse_labels[targets]


def get_cifar100superclass(data_dir, train_transform, test_transform, val_size, random_train_subset=False):
    trainset = CIFAR100(root=data_dir, transform=train_transform, train=True)
    test = CIFAR100(train=False, root=data_dir, transform=test_transform)

    if not random_train_subset:
        val, train = split_dataset(trainset, val_size)
    else:
        val, train = random_split(trainset, [val_size, len(trainset) - val_size])

    return CIFAR100SuperclassDataset(train), CIFAR100SuperclassDataset(val), CIFAR100SuperclassDataset(test)
