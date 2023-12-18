"""
Multi-taxonomy task heads, loss functions, and metrics.

Consider a set of labels grouped into a taxonomy of coarser labels.
There can be different but still-valid taxonomies.

For example, consider the following labels and taxonomies.
Labels -> apple, orange, beet, carrot
Color -> {apple, beet}, {orange, carrot}
Fruit -> {apple, orange}, {beet, carrot}
"""
from typing import Callable, Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader


def invert_taxonomy(taxonomy: Dict[int, List[int]]) -> Dict[int, int]:
    """Invert a "taxonomy group to members" map.

    Args:
        taxonomy (Dict[int, List[int]]): taxonomy group to member labels map

    Returns:
        Dict[int, int]: labels to taxonomy groupmap
    """
    inverse = {}
    for group, group_labels in taxonomy.items():
        for label in group_labels:
            inverse[label] = group
    return inverse


TAXONOMY_A = {
    0: [0],
    1: [1, 9],
    2: [2, 3, 4, 5, 6, 7, 8],
}
LABEL_TO_GROUP_A = invert_taxonomy(TAXONOMY_A)


def apply_taxonomy(
    labels: torch.Tensor,
    taxonomy: Dict[int, List[int]]
) -> torch.Tensor:
    """Map labels to coarser labels according to an taxonomy.

    Args:
        labels (torch.Tensor): labels to map
        taxonomy (Dict[int, List[int]]): taxonomy group to member labels map

    Returns:
        torch.Tensor: mapped labels
    """
    inverse = invert_taxonomy(taxonomy)
    return torch.tensor([inverse[int(curr)] for curr in labels])


def gather_outputs(
    network: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 10,
    device: str = 'cuda:0'
) -> Tuple[List[torch.FloatTensor], torch.LongTensor]:
    """Run entire dataset through a network (no grad).

    Args:
        network (torch.nn.Module): _description_
        dataset (torch.utils.data.Dataset): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]: network outputs and labels
    """
    loader = DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for images, labels in loader:
            outputs: List[torch.Tensor] = network(images.to(device))
            all_outputs.append([outp.cpu() for outp in outputs])
            all_labels.append(labels)
    all_outputs = [torch.cat(curr) for curr in zip(*all_outputs)]
    all_labels = torch.cat(all_labels)
    return all_outputs, all_labels


class TaxonomyHead(nn.Module):
    """Task head for labels and multiple label taxonomies. """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        taxonomies: List[Dict[int, List[int]]]
    ) -> None:
        super().__init__()
        if isinstance(taxonomies, dict):
            taxonomies = [taxonomies]

        self.linear = nn.Linear(in_features, num_classes, bias=True)
        self.taxonomies = taxonomies  # TODO - make deepcopy
        self.adders = []
        for taxonomy in taxonomies:
            # convert taxonomy to a matrix that implements score-adding
            adder = torch.zeros(num_classes, len(taxonomy))
            for parent, children in taxonomy.items():
                adder[children, parent] = 1
            self.adders.append(adder)

        # sanity check
        for outp in self(torch.rand(10, in_features)):
            assert torch.allclose(outp.sum(dim=1), torch.tensor(1.))

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Compute class confidence scores and taxonomy confidence scores.

        Args:
            features (torch.Tensor): feature vectors from some backbone (B, C)

        Returns:
            List[torch.Tensor]: class and taxonomy confidence scores
        """
        scores = self.linear(features).softmax(dim=1)
        outputs = [scores]
        for adder in self.adders:
            outputs.append(scores @ adder.to(scores.device))
        return outputs

    def compute_loss(
        self,
        outputs: List[torch.Tensor],
        labels: torch.Tensor,
        focal_gamma: float = None,
        label_smoothing: float = None
    ) -> List[torch.Tensor]:
        """_summary_

        Args:
            outputs (List[torch.Tensor]): _description_
            labels (torch.Tensor): _description_
            focal_gamma (float, optional): _description_. Defaults to None.
            label_smoothing (float, optional): _description_. Defaults to None.

        Returns:
            List[torch.Tensor]: _description_
        """
        loss_func = ClassBalancedLoss(
            FocalEntropyWithSmoothing(focal_gamma, label_smoothing)
        )
        losses = []
        losses.append(loss_func(outputs[0], labels))
        for curr_output, taxonomy in zip(outputs[1:], self.taxonomies):
            losses.append(
                loss_func(
                    curr_output,
                    apply_taxonomy(labels, taxonomy)
                )
            )
        return losses

    def compute_metrics(
        self,
        outputs: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Foo. """
        acc: List[float] = []
        pred = outputs[0].argmax(dim=1)
        acc.append(100 * (pred == labels).float().mean().item())
        for outp, taxonomy in zip(outputs[1:], self.taxonomies):
            pred = outp.argmax(dim=1)
            taxo_labels = apply_taxonomy(labels, taxonomy)
            acc.append(100 * (pred == taxo_labels).float().mean().item())
        return {
            f'acc/{ii}': float(acc)
            for ii, acc in enumerate(acc)
        }


class FocalEntropyWithSmoothing:
    """Cross-entropy with focal loss envelopes and label smoothing. """
    def __init__(
        self,
        focal_gamma: float = None,
        label_smoothing: float = None
    ) -> None:
        self.focal = focal_gamma
        self.smooth = label_smoothing

    def __repr__(self) -> str:
        return f'FocalEntropyWithSmoothing({self.focal}, {self.smooth})'

    def __call__(
        self,
        soft: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy.

        Args:
            soft (torch.Tensor): class confidence score.
            labels (torch.Tensor): true class labels

        Returns:
            torch.Tensor: cross-entropy
        """
        assert soft.ndim == 2
        assert labels.ndim == 1

        # apply focal envelope
        logsoft = soft.log()
        if self.focal is not None:
            logsoft = (1 - soft) ** self.focal * logsoft

        # apply label smoothing
        onehot = torch.eye(soft.shape[1], device=soft.device)[labels]
        if self.smooth is not None:
            onehot = (1 - self.smooth) * onehot
            onehot = onehot + self.smooth / soft.shape[1]

        return -(onehot * logsoft).sum(dim=1).mean()


class ClassBalancedLoss:
    """Utility class that averages loss by class not by sample. """
    def __init__(self, dependency: Callable) -> None:
        self.loss_func = dependency

    def __repr__(self) -> str:
        return f'ClassBalancedLoss({self.loss_func}))'

    def __call__(
        self,
        output: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss averaged by class.

        Args:
            output (torch.Tensor): network output
            labels (torch.Tensor): class labels

        Returns:
            torch.Tensor: loss
        """
        per_class_loss = []
        for curr_label in labels.unique():
            mask = labels == curr_label
            per_class_loss.append(self.loss_func(output[mask], labels[mask]))
        return torch.stack(per_class_loss).mean()
