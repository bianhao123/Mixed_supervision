__author__="Hao Bian"

import argparse
import random
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
import pandas as pd
import yaml
from addict import Dict
from pathlib import Path
import pprint
from mmcv import Config
import sys
import os.path as osp
# print(sys.path)
parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)

from datasets import DataInterface
from callbacks.common_callbacks import load_callbacks, load_loggers
from utils.util import  update_config, dynamic_import_from
from utils.config import get_config_import_model
import torch
from loguru import logger
from omegaconf import OmegaConf
# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import os



def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='train',
                        help='trainning stage or testing stage')
    parser.add_argument('--config', default='configs/SICAPv2.yaml',type=str)
    
    parser.add_argument('--gpus', default='1', type=str)
    parser.add_argument('--fold', default=1, type=int)


    parser.add_argument('--work_dir', default='./work_dir', type=str)

    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args = parser.parse_args()

    return args




def main(args, cfg):
    # 1 Initialize seed
    pl.seed_everything(cfg.General.seed)

    # 2 Define data interface and model interface
    dm = DataInterface(cfg)

    ModelInterface = dynamic_import_from(f'{cfg.Package_name.interface_name}', 'ModelInterface')
    model = ModelInterface(cfg)

    # 3 load callbacks, loggers
    loggers = load_loggers(cfg, logger_name='tensorboard')
    callbacks = load_callbacks(cfg)

    # 4 Instantiate Trainer
    trainer = Trainer(
        logger=loggers,
        log_every_n_steps=1,
        max_epochs= cfg.General.max_epochs,
        callbacks=callbacks,
        gpus=1,
        auto_select_gpus=True,
        # amp_level=cfg.General.amp_level,  # optimization level
        # precision=cfg.General.precision,  # mixed precision training
        benchmark=True,
        # accelerator='auto',
        accumulate_grad_batches=cfg.General.grad_acc, # 顺序算每个batch
        # deterministic=True,
        # weights_summary="full",
        num_sanity_val_steps=0,
        # overfit_batches=10,
        # check_val_every_n_epoch=1, 
        # fast_dev_run =5, # fast debug
        # resume_from_checkpoint = ,
        # limit_train_batches=0.01,
        # stochastic_weight_avg=False # SWA
    )

    # 5 train or test
    if args.stage == 'train':
        print('stage: train')
        trainer.fit(model=model, datamodule=dm)
        print('best_model_path:', cfg.callbacks[0].best_model_path)
        print('best_model_score', cfg.callbacks[0].best_model_score)
    elif args.stage == 'test':
        print('stage: test')
        model_paths = list(Path(cfg.log_path).glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg, strict=False)
            trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':

    # 1 load args
    args = make_parse()
    pprint.pprint(args)

    # 2 load config from yaml file
    cfg = get_config_import_model(args)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = Dict(cfg)
    # pprint.pprint(cfg)

    # 3 update the args to cfg
    update_config(args, cfg)

    main(args, cfg)
 