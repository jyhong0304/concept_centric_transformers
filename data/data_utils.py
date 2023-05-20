import numpy as np
from torch.utils.data import Dataset, Subset


class ExplanationDataset(Dataset):
    """
    An Abstract class for datasets with explanations
    """

    def get_data(self, idx):
        raise NotImplementedError

    def get_explanations(self, idx):
        # If not overloaded, return nan
        return np.nan

    def get_spatial_explanations(self, idx):
        # If not overloaded, return nan
        return np.nan

    def __getitem__(self, idx):
        inputs, labels = self.get_data(idx)
        return (inputs,
                self.get_explanations(idx),
                self.get_spatial_explanations(idx),
                labels)


def split_dataset(dataset, n_samples):
    """
    Splits a dataset in two: a first dataset consisting of the first `n_samples` samples, and the second consisting of the remaining ones
    """
    assert n_samples <= len(dataset), "The length of dataset has to be shorter than n_samples"
    sub_idx_1 = list(range(n_samples))
    sub_idx_2 = list(range(n_samples, len(dataset)))

    return Subset(dataset, sub_idx_1), Subset(dataset, sub_idx_2)
