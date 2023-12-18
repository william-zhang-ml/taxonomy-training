"""Training script. """
from typing import Tuple
import fire
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import yaml
from core import reader, utils


def get_new_run(
    config_yaml: str,
    tag: str = None,
    hparam_csv: str = None,
    hparam_row: int = None
) -> Tuple[dict, utils.Output]:
    """Set up a new training run.

    Args:
        config_yaml (str): path to base config file
        tag (str, optional): name of training run
        hparam_csv (str, optional): path to alternative/extra hparameters
        hparam_row (int, optional): which hparameter table row to use

    Returns:
        Tuple[dict, utils.Output]: training configuration and output directory
    """
    # initialize new output directory
    tag = 'default' if tag is None else tag  # default tag
    output = utils.Output('runs', tag)

    # load training run config and hparam substitutions
    with open(config_yaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if hparam_csv is not None and hparam_row is not None:
        config.update(utils.get_hyperparameters(hparam_csv, int(hparam_row)))
    with open(output.config_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file)

    return config, output


def main(
    config_yaml: str,
    tag: str = None,
    hparam_csv: str = None,
    hparam_row: int = None
) -> None:
    """Main training script.

    Args:
        config_yaml (str): path to base config file for new run
        tag (str, optional): name of new training run
        hparam_csv (str, optional): path to alternative/extra hparameters
        hparam_row (int, optional): which hparameter table row to use
    """
    config, output = get_new_run(config_yaml, tag, hparam_csv, hparam_row)

    print('Set up training variables ...')
    init_epoch = 0
    trainset = reader.DirectoryReader(
        './cifar10-train',
        transform=transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])
    )
    loader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    network = models.resnet18().to(config['device'])
    criteria = CrossEntropyLoss()
    config['optimizer']['kwargs']['params'] = network.parameters()
    optimizer = getattr(optim, config['optimizer']['name'])
    optimizer = optimizer(**config['optimizer']['kwargs'])
    config['scheduler']['kwargs']['optimizer'] = optimizer
    scheduler = getattr(optim.lr_scheduler, config['scheduler']['name'])
    scheduler = scheduler(**config['scheduler']['kwargs'])

    print('Training ...')
    epochbar = tqdm(
        range(init_epoch, config['num_epochs']),
        initial=init_epoch,
        total=config['num_epochs']
    )
    for i_epoch in epochbar:
        for images, labels in loader:
            logits = network(images.to(config['device']))
            loss = criteria(logits, labels.to(config['device']))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update feedback and logs
            epochbar.set_postfix({
                'loss': float(loss)
            })

            if 'plumbing' in config and config['plumbing']:
                print('... exit batch loop early for plumbing check')
                break

        scheduler.step()

        # save epoch checkpoint
        torch.save(
            {
                'epoch': i_epoch + 1,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            output.checkpoint_dir / f'{i_epoch + 1}.pt'
        )

        if 'plumbing' in config and config['plumbing']:
            print('... exit epoch loop early for plumbing check')
            break

    print('Exporting final weights ...')
    network.eval()
    torch.onnx.export(
        network,
        images.to(config['device']),
        output.onnx_path,
        input_names=['input'],
        output_names=['output']
    )

    print('Done!')
    output.write_done_token()


if __name__ == '__main__':
    fire.Fire(main)
