"""
This module contains a class that interfaces to a directory of JPG images.
The directory should contain subdirectories with INTEGER names.
Images should go into the subdirectory named after its label.
"""
import os
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset


class DirectoryReader(Dataset):
    """Interface to a directory of labelled JPG images. """
    def __init__(
        self,
        data_dir: str,
        transform: Callable = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_subdirs: List[Path] = []
        self.lookup: List[dict] = []

        # build a sample lookup table by looping over label sibdirectories
        for subdir in sorted(os.listdir(self.data_dir)):
            self.label_subdirs.append(self.data_dir / subdir)

            for filename in sorted(self.label_subdirs[-1].glob('*.jpg')):
                self.lookup.append({
                    'img_path': filename,
                    'label': int(subdir)
                })

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        lookup_entry = self.lookup[idx]
        img = plt.imread(lookup_entry['img_path'])
        if img.ndim == 2:
            img = img.unsqueeze(0)  # add channel dim to grayscale image
        elif img.ndim == 3:
            img = np.moveaxis(img, -1, 0)  # HWC -> CHW
        img = torch.from_numpy(img / 255)
        img = img.float()  # to float32 if float64
        if self.transform is not None:
            img = self.transform(img)
        label = lookup_entry['label']
        return img, label
