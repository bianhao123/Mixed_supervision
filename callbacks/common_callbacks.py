import argparse
from gc import callbacks
from numpy.core.arrayprint import DatetimeFormat
import yaml
from addict import Dict
from pathlib import Path
import pprint
# from experiment.models.model_interface import ModelInterface

import sys
import importlib
from utils.util import dynamic_import_from

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger
import datetime


def load_callbacks(cfg):
    """ defining all callbacks, including modelcheckpoint, earlystopping, and customize callback.

    Args:
        cfg (dict): Config parameters.

    Returns:
        List: List of all callbacks.
    """
    callbacks = []
    # add modelcheckpoint callback
    callbacks.append(ModelCheckpoint(monitor=cfg.Monitor.name,
                                     dirpath=str(cfg.log_path),
                                     filename=cfg.Monitor.filename,
                                     verbose=cfg.Monitor.verbose,
                                     save_last=True,
                                     save_top_k=cfg.Monitor.save_top_k,
                                     mode=cfg.Monitor.mode,
                                     save_weights_only=True))

    # add earlystopping callback
    callbacks.append(EarlyStopping(
        monitor='val_AUROC',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='max'
    ))

    # add customize callbacks
    for callback_info in cfg.Callback.values():
        callbacks.append(dynamic_import_from(
            callback_info[0], callback_info[1])())

    return callbacks


def load_loggers(cfg, logger_name='tensorboard'):

    logger = []
    if logger_name == 'tensorboard':
        log_path = cfg.General.log_dir
        Path(log_path).mkdir(exist_ok=True, parents=True)
        log_name = Path(cfg.config).parts[-2]  # 上一级地址
        version_name = Path(cfg.config).name[:-5]
        log_path = Path(log_path) / log_name / \
            version_name / f'fold{cfg.Data.fold}'
        cfg.log_path = log_path
        print(f'---->Log dir: {cfg.log_path}')

        # ---->TensorBoard
        tb_logger = pl_loggers.TensorBoardLogger(Path(cfg.General.log_dir) / log_name,
                                                 name=version_name, version=f'fold{cfg.Data.fold}',
                                                 log_graph=True, default_hp_metric=False)
        logger.append(tb_logger)
        # ---->CSV
        csv_logger = pl_loggers.CSVLogger(Path(cfg.General.log_dir) / log_name,
                                          name=version_name, version=f'fold{cfg.Data.fold}', )
        logger.append(csv_logger)

    # Wandb Logger
    if logger_name == 'wandb':
        if cfg.stage == 'test':
            cfg.General.wandb_project = cfg.General.wandb_project + '_test'
        logger = WandbLogger(name=cfg.General.wandb_name, project=cfg.General.wandb_project,
                             save_dir=cfg.work_dir, config=cfg, tags=cfg.tags)

    return logger
