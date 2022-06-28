import os.path as osp
import sys
from addict import Dict

parentdir = osp.dirname(osp.dirname(__file__))
sys.path.insert(0, parentdir)
from models.builder import build_model
from utils.config import load_config
import torch
from utils.util import dynamic_import_from

config_name = 'configs/SICAPv2.yaml'
cfg = load_config(config_name)
a = dynamic_import_from(f'models.{cfg.Model.type}', f'{cfg.Model.type}')

model = build_model(dict(cfg.Model))

data = torch.randn(1, 4, 1280)

output = model(data)
print(output)
print(output[0].shape)
print(output[1].shape)
pass
