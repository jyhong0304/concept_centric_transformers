from torch.utils.data import random_split, Dataset
from .data_utils import split_dataset, ExplanationDataset
import os
import numpy as np
from PIL import Image
import torch


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return sorted(dirs)
        else:
            return sorted(file)


class MakeImage():
    """
    this class used to make list of data for ImageNet
    """

    def __init__(self, data_dir, num_classes=200):
        self.image_root = data_dir
        self.category = get_name(os.path.join(data_dir, 'train'))
        self.used_cat = self.category[:num_classes]

    def get_data(self):
        train = self.get_img(self.used_cat, "train")
        val = self.get_img(self.used_cat, "val")
        return train, val, self.used_cat

    def get_img(self, folders, phase):
        record = []
        for folder in folders:
            current_root = os.path.join(self.image_root, phase, folder)
            images = get_name(current_root, mode_folder=False)
            for img in images:
                record.append([os.path.join(current_root, img), self.deal_label(folder)])
        return record

    def deal_label(self, img_name):
        back = self.used_cat.index(img_name)
        return back


class ImageNetDataset(Dataset):
    def __init__(self, data_dir, phase, transform=None):
        self.train, self.val, self.category = MakeImage(data_dir).get_data()
        self.all_data = {"train": self.train, "val": self.val}[phase]
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_root = self.all_data[item_id][0]
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.all_data[item_id][1]
        label = torch.from_numpy(np.array(label))
        # There is no global/spatial concepts.
        return image, np.nan, np.nan, label


def get_imagenet(data_dir, train_transform, test_transform, val_size, random_train_subset=False):
    trainset = ImageNetDataset(data_dir, "train", transform=train_transform)
    test = ImageNetDataset(data_dir, "val", transform=test_transform)

    if not random_train_subset:
        val, train = split_dataset(trainset, val_size)
    else:
        val, train = random_split(trainset, [val_size, len(trainset) - val_size])

    return train, val, test
