from typing import Optional, List


from pytorch_lightning.callbacks import Callback
import copy
import torch
from omegaconf import OmegaConf
from utils.util import dynamic_import_from

from loss.common_loss import get_loss


class CombinedCriterion(torch.nn.Module):
    def __init__(self, loss: dict, device) -> None:
        super().__init__()
        self.slide_criterion = get_loss(loss, "slide", device)
        self.instance_criterion = get_loss(loss, "instance", device)
        self.instance_loss_weight = loss.get(
            "params", {}).get("instance_weight", 0.5)
        assert (
            0.0 <= self.instance_loss_weight <= 1.0
        ), f"instance weight loss must be between 0 and 1, but is {self.instance_loss_weight}"
        self.slide_loss_weight = 1.0 - self.instance_loss_weight
        self.device = device

    def forward(
        self,
        slide_logits: Optional[torch.Tensor] = None,
        slide_labels: Optional[torch.Tensor] = None,
        instance_logits: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
        instance_associations: Optional[List[int]] = None,
        drop_slide: Optional[bool] = False,
        drop_instance: Optional[bool] = False,
    ):
        assert (
            slide_logits is not None and slide_labels is not None
        ), "Cannot use combined criterion without slide input"
        assert (
            instance_logits is not None and instance_labels is not None
        ), "Cannot use combined criterion without instance input"
        instance_labels = instance_labels.to(self.device)
        slide_labels = slide_labels.to(self.device)

        slide_loss = self.slide_criterion(
            logits=slide_logits,
            targets=slide_labels
        )
        instance_loss = self.instance_criterion(
            logits=instance_logits,
            targets=instance_labels,
            instance_associations=instance_associations,
        )

        if drop_slide:
            combined_loss = self.slide_loss_weight * 0 + \
                self.instance_loss_weight * instance_loss
        elif drop_instance:
            combined_loss = self.slide_loss_weight * \
                slide_loss + self.instance_loss_weight * 0
        else:
            combined_loss = (
                self.slide_loss_weight * slide_loss + self.instance_loss_weight * instance_loss
            )
        # print('only slide loss')
        # combined_loss = slide_loss
        return combined_loss, slide_loss.detach().cpu(), instance_loss.detach().cpu()


class mixed_loss(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Starting to init trainer!")
        params = pl_module.cfg
        train_loss = copy.deepcopy(params["Loss"])
        # if type(train_loss) != dict:
        #     train_loss = OmegaConf.to_container(train_loss, resolve=True)
        train_dataset = trainer.datamodule.train_dataset
        device = pl_module.device
        if train_loss["params"]["use_weighted_loss"]:
            train_dataset.set_mode("instance")
            train_loss["instance"]["params"]["weight"] = train_dataset.get_dataset_loss_weights(
                log=train_loss["params"]["use_log_frequency_weights"]
            )
            train_dataset.set_mode("slide")
            train_loss["slide"]["params"]["weight"] = train_dataset.get_dataset_loss_weights(
                log=train_loss["params"]["use_log_frequency_weights"]
            )
            train_dataset.set_mode("instance")
            pl_module.train_criterion = CombinedCriterion(train_loss, device)

        val_dataset = trainer.datamodule.val_dataset
        val_loss = copy.deepcopy(params["Loss"])
        # if type(val_loss) != dict:
        #     val_loss = OmegaConf.to_container(val_loss, resolve=True)
        if val_loss["params"]["use_weighted_loss"]:
            val_dataset.set_mode("instance")
            val_loss["instance"]["params"]["weight"] = val_dataset.get_dataset_loss_weights(
                log=val_loss["params"]["use_log_frequency_weights"]
            )
            val_dataset.set_mode("slide")
            val_loss["slide"]["params"]["weight"] = val_dataset.get_dataset_loss_weights(
                log=val_loss["params"]["use_log_frequency_weights"]
            )
            val_dataset.set_mode("instance")
        pl_module.val_criterion = CombinedCriterion(val_loss, device)

    # def on_init_end(self, trainer):
    #     print("trainer is init now")

    # def on_train_end(self, trainer, pl_module):
    #     print("do something when training ends")


# trainer = Trainer(callbacks=[MyPrintingCallback()])
