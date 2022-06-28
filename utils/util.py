import importlib
from typing import Optional, Any, Union
import yaml
from addict import Dict
import os
import torch.nn.functional as F
import torch


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)


def load_single_PLModel(config_path: str, model_path: str):
    with open(config_path, mode="r") as file:
        cfg = Dict(yaml.load(file, Loader=yaml.Loader))

    ModelInterface = getattr(importlib.import_module(
        f'models.{cfg.Model.interface}'), 'ModelInterface')
    model = ModelInterface(cfg)
    # ,map_location={'cuda:1':'cuda:0'}
    model = model.load_from_checkpoint(
        checkpoint_path=model_path, cfg=cfg, strict=False)
    # model = model.cuda()
    return model


def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def update_config(args, cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # if args.gpus is not None:
    # nums_gpus = len(args.gpus.split(','))
    # cfg.General.gpus = ','.join([str(i) for i in range(nums_gpus)])

    # if args.test_batch_size is not None:
    #     cfg.Data.test_dataset.batch_size = args.test_batch_size
    if args.fold is not None:
        cfg.Data.fold = args.fold

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.config is not None:
        cfg.config = args.config

    # if args.checkpoint_path is not None:
    #     cfg.checkpoint_path = args.checkpoint_path

    # if args.debug is not None:
    #     cfg.debug = args.debug

    cfg.stage = args.stage
    # os.environ['CUDA_VISIBLE_DEVICES']=cfg.General.gpus
    # cfg.tags = args.tags


def update_sweep_config(args, cfg):

    # if args.grad_acc is not None:
    #     cfg.General.grad_acc = args.grad_acc
    #     print(f'sweeping grad_acc: {cfg.General.grad_acc}')
    # if args.instance_weight is not None:
    #     cfg.train.params.loss.params.instance_weight = args.instance_weight
    #     print(f'sweeping instance_weight: {cfg.train.params.loss.params.instance_weight}')

    # if args.seed is not None:
    #     cfg.General.seed = args.seed
    #     print(f'sweeping seed: {cfg.General.seed}')
    # print('')
    pass


def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=0) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]])
                         for i in range(len(y))])
    loss = - torch.sum(x_log) / len(y)
    return loss
