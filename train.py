"""
Training script.

Notes on CIFAR10 labels.
0 -> airplane
1 -> automobile
2 -> bird
3 -> cat
4 -> deer
5 -> dog
6 -> frog
7 -> horse
8 -> ship
9 -> truck

Class-balanced taxonomy.
Airplane -> 0 ... 100 samples
Not-plane vehicles -> 1, 9 ... 60 samples
Animals -> 2, 3, 4, 5, 6, 7, 8 ... 70 samples
"""
from pathlib import Path
from typing import List, Tuple, Union
import fire
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import yaml
from core import reader, taxonomy, utils


def get_new_run(
    config_yaml: Union[str, Path],
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
        Tuple[dict, utils.output]: training configuration and output directory
    """
    # initialize new output directory
    tag = 'default' if tag is None else tag  # default tag
    output = utils.Output('runs', tag)

    # load training run config and hparam substitutions
    with open(config_yaml, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if hparam_csv is not None and hparam_row is not None:
        config.update(utils.get_hyperparameters(hparam_csv, int(hparam_row)))
    config['commithash'] = utils.get_repo_hash()
    with open(output.config_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config, file)

    return config, output


def load_checkpoint(tag: str) -> Tuple[dict, utils.Output, dict]:
    """Load existing output directory, its config, and the latest checkpoint.

    Args:
        tag (str): output directory tag

    Returns:
        Tuple[dict, utils.output, dict]:
            config,
            output directory interface,
            checkpoint state
    """
    output = utils.Output('runs', tag, exists=True)
    with open(output.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    checkpoint = output.get_latest_checkpoint()
    return config, output, checkpoint


def main(
    run_arg: str,
    tag: str = None,
    hparam_csv: str = None,
    hparam_row: int = None
) -> None:
    """Main training script.

    Args:
        run_arg (str): path to base config file for new run,
                       or to existing run's output directory
        tag (str, optional): name of new training run
        hparam_csv (str, optional): path to alternative/extra hparameters
        hparam_row (int, optional): which hparameter table row to use
    """

    print('Load config ...')
    run_arg = Path(run_arg)
    if run_arg.is_file():
        config, output = get_new_run(run_arg, tag, hparam_csv, hparam_row)
        checkpoint = None
    elif run_arg.is_dir():
        print('... detected checkpoint, loading from checkpoint')
        config, output, checkpoint = load_checkpoint(run_arg.stem)
        print(f'... will start training from epoch {checkpoint["epoch"]}')
    else:
        raise FileNotFoundError('must pass config yaml or prev output dir')

    print(f'Linking to tensorboard w/tag "{output.tag}" ...')
    board = SummaryWriter(log_dir=f'_tensorboard/{output.tag}')
    val_board = SummaryWriter(log_dir=f'_tensorboard/{output.tag}-val')

    print('Set up training variables ...')
    init_epoch, step = 0, 0
    trainset = reader.DirectoryReader(
        './cifar10-train',
        transform=transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])
    )
    valset = CIFAR10(
        './cifar10-train',
        train=False,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    network = models.resnet18().to(config['device'])
    head = taxonomy.TaxonomyHead(
        network.fc.in_features,
        10,
        taxonomy.TAXONOMY_A
    )
    network.fc = head
    config['optimizer']['kwargs']['params'] = network.parameters()
    optimizer = getattr(optim, config['optimizer']['name'])
    optimizer = optimizer(**config['optimizer']['kwargs'])
    config['scheduler']['kwargs']['optimizer'] = optimizer
    scheduler = getattr(optim.lr_scheduler, config['scheduler']['name'])
    scheduler = scheduler(**config['scheduler']['kwargs'])

    if checkpoint is not None:
        print('... detected checkpoint, restoring state')
        init_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        network.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print('Training ...')
    epochbar = tqdm(
        range(init_epoch, config['num_epochs']),
        initial=init_epoch,
        total=config['num_epochs']
    )
    val_acc, val_loss = float('nan'), float('nan')
    for i_epoch in epochbar:
        for images, labels in loader:
            step += 1

            # forward, backward, metrics
            outputs: List[torch.Tensor] = network(images.to(config['device']))
            losses = head.compute_loss(
                outputs,
                labels.to(config['device']),
                2,
                0.5
            )
            loss = losses[0] + 0.5 * losses[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metrics = head.compute_metrics(outputs, labels)

            # update feedback and logs
            for i_loss, value in enumerate(losses):
                board.add_scalar(f'loss/{i_loss}', value, step)
            for name, value in metrics.items():
                board.add_scalar(name, value, step)
            epochbar.set_postfix({
                'acc': float(metrics['acc/0']),  # full label set
                'loss': float(loss),
                'val_acc': float(val_acc),
                'val_loss': float(val_loss)
            })

            if 'plumbing' in config and config['plumbing']:
                print('... exit batch loop early for plumbing check')
                break

        scheduler.step()

        # save epoch checkpoint
        torch.save(
            {
                'epoch': i_epoch,
                'step': step,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            output.checkpoint_dir / f'{i_epoch + 1}.pt'
        )

        # compute current weight validation metrics
        network.eval()
        with torch.no_grad():
            outputs, labels = taxonomy.gather_outputs(
                network,
                valset,
                device=config['device'],
                num_batches=100
            )
            losses = head.compute_loss(outputs, labels, 2, 0.5)
            val_loss = losses[0] + 0.5 * losses[1]
            val_metrics = head.compute_metrics(outputs, labels)
            val_acc = val_metrics['acc/0']
        network.train()
        for i_loss, value in enumerate(losses):
            val_board.add_scalar(f'loss/{i_loss}', value, step)
        for name, value in val_metrics.items():
            val_board.add_scalar(name, value, step)

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
