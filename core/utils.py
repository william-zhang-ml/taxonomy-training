"""Utility functions and classes. """
import os
from pathlib import Path
import pandas as pd
import torch


def get_hyperparameters(file: str, index: int) -> dict:
    """Get a specific hyperparameter row from a CSV table.

    Args:
        file (str): CSV table of hyperparmeters
        index (int): row to get

    Returns:
        dict: hyperparameter key-value pairs
    """
    hparams = {}
    for key, val in pd.read_csv(file).loc[index].dropna().to_dict().items():
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        hparams[key] = val
    return hparams


class Output:
    """Training run output directory interface."""
    def __init__(self, root, tag, exists: bool = False):
        self.root = Path(root)
        self.tag = tag
        os.makedirs(self.checkpoint_dir, exist_ok=exists)

    @property
    def output_dir(self) -> Path:
        """
        Returns:
            Path: path to outputs
        """
        return self.root / self.tag

    @property
    def checkpoint_dir(self) -> Path:
        """
        Returns:
            Path: path to training checkpoints
        """
        return self.output_dir / 'checkpoints'

    @property
    def config_path(self) -> Path:
        """
        Returns:
            Path: path to config file used to initiate training
        """
        return self.output_dir / 'config.yaml'

    @property
    def onnx_path(self) -> Path:
        """
        Returns:
            Path: path to final weights
        """
        return self.output_dir / 'final.onnx'

    def get_latest_checkpoint(self) -> dict:
        """Load most-recent (alphabetical) training checkpoint's state.

        Returns:
            dict: checkpoint state
        """
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'))
        return torch.load(checkpoints[-1])

    def write_done_token(self) -> None:
        """Write 'done.txt' to output directory to show training run done. """
        token_path = self.output_dir / 'done.txt'
        with open(token_path, 'w', encoding='utf-8') as file:
            file.write('done')

    def is_done(self) -> bool:
        """Check whether a training run is done (via token).

        Returns:
            bool: whether 'done.txt' exists in the output directory
        """
        return os.path.isfile(self.output_dir / 'done.txt')
