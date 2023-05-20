import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from .cifar100superclass_dataset import get_cifar100superclass


class CIFAR100SuperClass(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 data_dir: str = '~/datasets',
                 val_size: int = 5000,
                 **kwargs
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.num_classes = 100
        self.num_super_classes = 20

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        trainset, valset, testset = get_cifar100superclass(data_dir=self.data_dir, train_transform=self.train_transform,
                                                           test_transform=self.test_transform, val_size=self.val_size)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cifar100_val, self.cifar100_train = valset, trainset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar100_test = testset

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    data_dir = '~/datasets'
    explanation = CIFAR100SuperClass(data_dir=data_dir)
    explanation.prepare_data()
    explanation.setup()
    train_dl = explanation.train_dataloader()
    val_dl = explanation.val_dataloader()
    test_dl = explanation.test_dataloader()
    print(f"Dataset split (train/val/test): {len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")
