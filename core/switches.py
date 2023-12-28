"""
Metaphorical switchboard module for instantiating training objects.

This module lets the main script mix and match networks/optimizers/etc,
This module hides instantiation differences.
The main script can switch options (ex: resnet18) per category (ex: network).
"""
from torch import nn
from torchvision import models


def get_backbone(arch_name: str, task_head: nn.Module) -> nn.Module:
    """Affix a task head to a CNN backbone archiecture.

    Args:
        arch_name (str): name of backbone
        task_head (nn.Module): any CNN-compatible task head

    Raises:
        ValueError: backbone name lookup error

    Returns:
        nn.Module: final network for training
    """
    if arch_name == 'resnet18':
        network = models.resnet18()
        network.fc = task_head
    elif arch_name == 'resnext32':
        network = models.resnext50_32x4d()
        network.fc = task_head
    else:
        raise ValueError(f'arch_name "{arch_name}" is not supported')
    return network
