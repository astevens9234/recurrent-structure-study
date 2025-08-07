"""Data Classes"""

import os

import pandas as pd
import torch


class MouseDataset(torch.utils.data.Dataset):
    """Dataset of mouse movements captured from mouse_listener.py"""

    def __init__(self, data_dir="../data", transform=None):
        """
        Args:
            data_dir: location of .csv files
        """
        self.data_dir = data_dir
        self.getcwd = os.getcwd()
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pd.read_csv(self.data_dir + "/" + self.files[index])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class Sequence(object):
    """Extract individual mouse pathes from .csv files."""

    def __call__(self, sample):
        idx = list(sample[sample["action"] == "released"].index)
        pathes = []
        pathes.append(sample[0 : idx[1] + 1])

        for i in range(0, len(idx) - 1):
            pathes.append(sample[idx[i] + 1 : idx[i + 1] + 1])

        return pathes


class Wrangle(object):
    """Wrangling of individual pathes."""

    def __call__(self, sample):
        for j in range(0, len(sample)):
            ts_min = sample[j]["ts"].min()
            sample[j]["ts"] = sample[j]["ts"] - ts_min
            sample[j]["event"] = (
                sample[j]["event"].replace({"move": 0, "click": 1}).astype(int)
            )
            sample[j]["button"] = (
                sample[j]["button"]
                .replace({"Button.left": 1, "Button.right": 2})
                .fillna(0)
                .astype(int)
            )
            sample[j]["action"] = (
                sample[j]["action"]
                .replace({"press": 1, "released": 2})
                .fillna(0)
                .astype(int)
            )

        return sample


class ToTensor(object):
    """Convert sequences to tensors."""
    
    def __call__(self, sample):
        tensor_list = [torch.tensor(s.values) for s in sample]
        return tensor_list


class CompositeTransforms:
    """Wrapper class for all transformations. Pass in a list of transform classes on call.
    Passable to Dataset e.x. MouseData(transforms=CompositeTransforms([...]))
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
