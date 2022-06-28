__author__ = "Hao Bian"
import argparse
from functools import partial
import inspect
from pathlib import Path
from cv2 import phase

import pandas as pd
import sys
import os
import numpy as np
import importlib
import copy
from os.path import join as opj
import matplotlib.pyplot as plt
import openslide
from pytorch_lightning import loggers

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl
from torchmetrics import F1, ConfusionMatrix

from loss.loss_factory import create_loss, create_metric
from timm.optim.optim_factory import create_optimizer
from models import build_model

# to import loss
import torchmetrics
import random


class ModelInterface(pl.LightningModule):

    # init

    def __init__(self, cfg):
        super(ModelInterface, self).__init__()
        # self.save_hyperparameters()
        self.cfg = cfg
        self.load_model()  # load model
        self.configure_loss_metric()  # load metric

    def forward(self, *x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(*x)

    #*********train step*************************************************************#
    def training_step(self, batch, batch_idx):

        # load data of batch
        data_dict = batch
        in_features = data_dict['instance_feature']
        coords = data_dict['instance_centroid']
        slide_label, instance_label = data_dict['slide_label'], data_dict['instance_label'].squeeze(
        )

        # training forward
        slide_logits, instance_logits, index = self(
            in_features, coords, 'train')  # [1, C] [1, N, C]

        instance_logits = instance_logits.squeeze()
        instance_label = torch.index_select(
            instance_label, 0, index)  # masking the corresponding label

        loss_infomation = {
            'slide_logits': slide_logits,
            "slide_labels": slide_label,
            "instance_logits": instance_logits,
            "instance_labels": instance_label,
            "instance_associations": len(instance_label),
        }

        # compute the slide-level and instance-level loss
        combined_loss, slide_loss, instance_loss = self.train_criterion(
            **loss_infomation)

        return {'loss': combined_loss}

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs

        pass

    #*********validation step*************************************************************#

    def validation_step(self, batch, batch_idx):

        data_dict = batch
        in_features = batch['instance_feature']
        coords = data_dict['instance_centroid']
        instance_associations = len(data_dict['instance_label'])
        slide_label, instance_label = data_dict['slide_label'], data_dict['instance_label'].squeeze(
        )

        slide_logits, instance_logits = self(
            in_features, coords)  # [4, 4] [531, 4]

        instance_logits = instance_logits.squeeze()

        loss_information = {
            'slide_logits': slide_logits,
            "slide_labels": slide_label,
            "instance_logits": instance_logits,
            "instance_labels": instance_label,
            "instance_associations": instance_associations,
            # "drop_slide": drop_slide #
        }
        combined_loss, slide_loss, instance_loss = self.val_criterion(
            **loss_information)

        return {'val_loss': combined_loss, 'slide_logits': loss_information['slide_logits'].detach(), 'slide_labels': loss_information["slide_labels"].detach(), 'instance_logits': loss_information['instance_logits'].detach().cpu(), 'instance_labels': loss_information['instance_labels'].detach().cpu()}

    def validation_epoch_end(self, val_step_outputs):

        combined_loss, slide_logits, slide_labels, instance_logits, instance_labels = list(
            zip(*[i.values() for i in val_step_outputs]))

        # * cls metric
        slide_logits = torch.cat(slide_logits)
        slide_labels = torch.cat(slide_labels)
        val_loss = torch.stack(combined_loss)

        self.log('val_loss', val_loss.mean(),
                 prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics.cpu()(slide_logits.cpu(), slide_labels.cpu()), prog_bar=True,
                      on_epoch=True, logger=True)

    def configure_optimizers(self):
        args = argparse.Namespace()
        for key, value in self.cfg.Optimizer.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        optimizer = create_optimizer(args, self.model)
        return optimizer

    def configure_loss_metric(self):
        # loss = self.cfg.Loss.baseloss

        # torch metric
        metrics = torchmetrics.MetricCollection([torchmetrics.AUROC(
            num_classes=self.cfg.Model.num_classes, average='macro')])
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    #*********test step*************************************************************#

    def test_step(self, batch, batch_idx):

        self.eval()
        return_dict = {}
        data_dict = batch
        in_features = batch['instance_feature']
        coords = data_dict['instance_centroid']
        instance_associations = len(data_dict['instance_label'])
        slide_label, instance_label = data_dict['slide_label'], data_dict['instance_label'].squeeze(
        )
        slide_logits, instance_logits = self(
            in_features, coords)  # [1, C] [1, N, C]

        instance_logits = instance_logits.squeeze()
        return_dict.update(slide_logits=slide_logits.cpu())
        return_dict.update(slide_labels=slide_label.cpu())

        return return_dict

    def test_epoch_end(self, test_step_outputs):

        slide_logits, slide_labels = list(
            zip(*[i.values() for i in test_step_outputs]))
        slide_logits = torch.cat(slide_logits)
        slide_labels = torch.cat(slide_labels)

        metrics = self.test_metrics.cpu()(slide_logits.cpu(), slide_labels.cpu())

        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()

        # Save all indicators into CSV
        result = pd.DataFrame([metrics])
        result.to_csv(self.cfg.log_path / 'result.csv')

    def load_model(self):
        self.model = build_model(self.cfg.Model)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
