import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .imagenet_dataset import get_imagenet
from PIL import Image


class ImageNetDatamodule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 data_dir: str = '~/datasets/',
                 val_size: int = 20000,
                 **kwargs
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.num_classes = 1000

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        trainset, valset, testset = get_imagenet(data_dir=self.data_dir, train_transform=self.train_transform,
                                                           test_transform=self.test_transform, val_size=self.val_size)

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.imagenet_val, self.imagenet_train = valset, trainset

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.imagenet_test = testset

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.imagenet_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    data_dir = '/data/jhong53/datasets/imagenet/'
    explanation = ImageNetDatamodule(data_dir=data_dir)
    explanation.prepare_data()
    explanation.setup()
    train_dl = explanation.train_dataloader()
    val_dl = explanation.val_dataloader()
    test_dl = explanation.test_dataloader()
    print(f"Dataset split (train/val/test): {len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")
