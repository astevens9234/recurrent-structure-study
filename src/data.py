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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return pd.read_csv(self.data_dir + "/" + self.files[index])
    

class Sequence(object):
    """Extract individual mouse pathes from .csv files."""
    ...


class ToTensor(object):
    """Convert sequences to tensors."""
    ...