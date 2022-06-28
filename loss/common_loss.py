from typing import List
import torch
from torch import nn
from utils.util import dynamic_import_from


class MultiLabelBCELoss(nn.Module):
    """Binary Cross Entropy loss over each label seperately, then averaged"""

    def __init__(self, weight=None) -> None:
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none" if weight is not None else "mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the logit and targets

        Args:
            logits (torch.Tensor): Logits for the slide with the shape: B x nr_classes
            targets (torch.Tensor): Targets one-hot encoded with the shape: B x nr_classes

        Returns:
            torch.Tensor: Slide loss
        """
        if self.weight is None:
            return self.bce(input=logits, target=targets.to(torch.float32))
        else:
            loss = self.bce(input=logits, target=targets.to(torch.float32))
            weighted_loss = loss * self.weight.to(loss.device)
            return weighted_loss.mean()


class InstanceStochasticCrossEntropy(nn.Module):
    def __init__(self, drop_probability=0.0, instances_to_keep=None, background_label=4, weight=None) -> None:
        super().__init__()
        assert (
            0.0 <= drop_probability <= 1.0
        ), f"drop_probability must be valid proability but is {drop_probability}"
        self.drop_probability = drop_probability
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=background_label, weight=weight)
        self.instances_to_keep = instances_to_keep

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, instance_associations: List[int]
    ) -> torch.Tensor:
        """Compute the loss of the given logits and target labels
        Args:
            logits (torch.Tensor): Logits for the instances with the shape: \sum_{i=0}^B nr_instances x nr_classes
            targets (torch.Tensor): Targets labels with the shape: \sum_{i=0}^B nr_instances
            slide_associations (List[int]): Information needed to unbatch logits and targets
        Returns:
            torch.Tensor: instance loss
        """
        if self.instances_to_keep is not None:
            to_keep_mask = list()
            start = 0
            for n in instance_associations:
                indices = torch.arange(start, start + n).to(torch.float32)
                num_samples = min(len(indices), self.instances_to_keep)
                samples = torch.multinomial(
                    torch.ones_like(indices), num_samples=num_samples)
                to_keep_mask.append(indices[samples].to(torch.int64))
                start += n
            to_keep_mask = torch.cat(to_keep_mask)
            targets = targets[to_keep_mask]
            logits = logits[to_keep_mask]
        elif self.drop_probability > 0:
            to_keep_mask = torch.rand(targets.shape[0]) > self.drop_probability
            targets = targets[to_keep_mask]
            logits = logits[to_keep_mask]
        targets = targets.long()
        return self.cross_entropy(logits.float(), targets)


def get_loss_criterion(loss, dataset, supervision_mode, name=None, device=None):
    if loss["params"]["use_weighted_loss"]:
        dataset.set_mode(supervision_mode)
        loss[name]["params"]["weight"] = dataset.get_dataset_loss_weights(
            log=loss["params"]["use_log_frequency_weights"]
        )
    return get_loss(loss, name, device)


def get_loss(config, name=None, device=None):
    if name is not None:
        config = config[name]
    # loss_class = dynamic_import_from("losses", config["class"])
    loss_class = dynamic_import_from(
        "loss.common_loss", config["class"])
    criterion = loss_class(**config.get("params", {}))
    return criterion.to(device) if device is not None else criterion


def get_optimizer(optimizer, model):
    optimizer_class = dynamic_import_from("torch.optim", optimizer["class"])
    optim = optimizer_class(model.parameters(), **optimizer["params"])

    # Learning rate scheduler
    scheduler_config = optimizer.get("scheduler", None)
    if scheduler_config is not None:
        scheduler_class = dynamic_import_from(
            "torch.optim.lr_scheduler", scheduler_config["class"]
        )
        scheduler = scheduler_class(
            optim, **scheduler_config.get("params", {}))
    else:
        scheduler = None
    return optim, scheduler
