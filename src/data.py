"""Data Classes"""

import os

import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder

pd.set_option("future.no_silent_downcasting", True)
pd.options.mode.chained_assignment = None


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

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.col_space = ["button", "action", "event"]
        self.all_cols = [
            "ts",
            "x",
            "y",
            "action_press",
            "action_released",
            "action_nan",
            "button_Button",
            "button_nan",
            "button_delta",
            "delta_x_coord",
            "delta_y_coord",
            "event_click",
            "event_scroll",
            "event_move",
        ]

    def __call__(self, sample):
        for j in range(0, len(sample)):
            ts_min = sample[j]["ts"].min()
            sample[j]["ts"] = sample[j]["ts"] - ts_min

            sample[j] = self._split_deltas(sample[j])
            sample[j] = self._encode(sample[j])

        return sample

    def _split_deltas(self, frame):
        splt = frame["button"].str.extract(r"\((-?\d+),(-?\d+)\)")
        splt.columns = ["delta_x_coord", "delta_y_coord"]
        frame["button"] = frame["button"].str.extract(r"(\w+)")[0]

        splt["delta_x_coord"] = splt["delta_x_coord"].fillna(0).astype(int)
        splt["delta_y_coord"] = splt["delta_y_coord"].fillna(0).astype(int)

        return pd.merge(frame, splt, how="left", on=frame.index).drop(columns=["key_0"])

    def _encode(self, frame):
        encoded_data = self.encoder.fit_transform(frame[self.col_space])
        result = pd.merge(
            frame.drop(columns=self.col_space),
            pd.DataFrame(
                encoded_data, columns=self.encoder.get_feature_names_out(self.col_space)
            ),
            on=frame.index,
            validate="1:1",
        ).drop(columns="key_0")

        # enforce column space
        missing_cols = [c for c in self.all_cols if c not in result]
        for col in missing_cols:
            result[col] = 0

        return result


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
