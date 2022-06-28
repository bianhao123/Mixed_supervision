__author__ = "Hao Bian"

import glob
import os
import sys
sys.path.append('.')
from tqdm import tqdm
from utils.util import read_yaml
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

import torch
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader

from loguru import logger

from utils.constants import Constants
from utils.metrics import inverse_frequency, inverse_log_frequency
from utils.utils_mixed import get_metadata, fast_histogram
from pathlib import Path
from tqdm import tqdm
import dgl
from dgl.data.utils import load_graphs
NR_CLASSES = 4


def to_mapper(df):
    """ map the raw label into vector

    Args:
        df (DataFrame): Record the dataframe of the data label
    Returns:
        dict: 
            E.g.: {'xxx': array([0,0,1,1]),
                    ...
                    }
    """
    mapper = dict()
    for name, row in df.iterrows():
        mapper[name] = np.array(
            [row["benign"], row["grade3"], row["grade4"], row["grade5"]]
        )
    return mapper


class MixedData(data.Dataset):
    """ dataset of mixed supervision pipeline

    Args:
        dataset_cfg (dict): Define from the config file(yaml).
        phase (str): 'train' or 'test'. If 'train', return the traindataset. If 'test', return the testdataset.

    """

    def __init__(self, dataset_cfg=None,
                 phase=None):
        # set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        self.num_classes = NR_CLASSES
        self.nfolds = self.dataset_cfg.nfolds
        self.fold = self.dataset_cfg.fold
        self.base_path = self.dataset_cfg.base_path
        self.csv_dir = self.dataset_cfg.label_dir + f'/fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        # load all sample info with tow data modes
        self.data_load_mode = self.dataset_cfg.data_load_mode
        if self.data_load_mode == 0:
            self.slide_paths = sorted(
                glob.glob(os.path.join(self.base_path, '*.bin')))
            self.slide_paths = {
                Path(slide_path).stem: slide_path for slide_path in self.slide_paths}
        else:
            constants = Constants(base_path=Path(self.base_path), mode=phase,
                                  fold=self.fold, partial=100)
            self.all_metadata, self.image_label_mapper = get_metadata(
                constants)

        # split the dataset
        if phase == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.data.index = range(len(self.data))

        if phase == 'valid':
            self.data = self.slide_data.loc[:, 'val'].dropna()
            self.data.index = range(len(self.data))

        if phase == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.data.index = range(len(self.data))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _load_name(i, row):
        return i

    def _load_slide(self, slide_id):
        row = self.all_metadata.loc[slide_id]
        slide = load_graphs(str(row["graph_path"]))[0][0]
        slide.readonly()
        return slide

    def _load_slide_label(self, slide_id):
        return self.image_label_mapper[slide_id]

    def _load_slide_and_label(self, slide_id):
        slide_path = self.slide_paths[slide_id]
        slide, label = load_graphs(slide_path)
        slide = slide[0]
        slide.readonly()
        return slide, label['slide_label']

    def set_mode(self, mode):
        valid_modes = ["slide", "instance"]
        assert (
            mode in valid_modes
        ), f"Dataset mode must be from {valid_modes}, but is {mode}"
        self.supervision_mode = mode

    def get_labels(self) -> torch.Tensor:
        if self.supervision_mode == "instance":
            instance_labels = list()
            nr_datapoints = self.__len__()
            for i in range(nr_datapoints):
                datapoint = self.__getitem__(i)
                instance_labels.append(datapoint['instance_label'])
            return torch.cat(instance_labels)

        elif self.supervision_mode == "slide":
            slide_labels = list()
            nr_datapoints = self.__len__()
            for i in range(nr_datapoints):
                datapoint = self.__getitem__(i)
                slide_labels.append(datapoint['slide_label'])
            if self.data_load_mode == 0:
                return torch.stack(slide_labels)
            else:
                return torch.from_numpy(np.array(slide_labels))
        else:
            raise NotImplementedError

    def get_dataset_loss_weights(self, log=True) -> torch.Tensor:

        if self.supervision_mode == "instance":
            instance_label_path = Path(
                self.dataset_cfg.label_dir) / f'{self.phase}_instance_label_fold{self.fold}.pt'
            if instance_label_path.exists():
                labels = torch.load(instance_label_path)
            else:
                labels = self.get_labels()
                torch.save(labels, instance_label_path)
            class_counts = fast_histogram(labels, self.num_classes)
        else:
            slide_label_path = Path(
                self.dataset_cfg.label_dir) / f'{self.phase}_slide_label_fold{self.fold}.pt'
            if slide_label_path.exists():
                labels = torch.load(slide_label_path)
            else:
                labels = self.get_labels()
                torch.save(labels, slide_label_path)
            class_counts = labels.sum(dim=0).numpy()
        if log:
            class_weights = inverse_log_frequency(
                class_counts.astype(np.float32)[np.newaxis, :]
            )[0]
        else:
            class_weights = inverse_frequency(
                class_counts.astype(np.float32)[np.newaxis, :]
            )[0]
        return torch.as_tensor(class_weights)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): random index from dataloader

        Returns:
            dict: Return features and labels of slide and instance. 
                    E.g.: {'slide_id': 16B0001851, 
                            'slide_label': torch.Tensor of shape C,
                            'instance_centroid': torch.Tensor of shape N x 2,
                            'instance_label': torch.Tensor of shape N,
                            'instance_feature': torch.Tensor of shape N x D
                            }
                    N: Number of instance.
                    C: Class number of prediction.
                    D: The dimension of instance feature.
        """
        data_dict = dict()  # {slide, slide_label, instance_label}
        slide_id = self.data[idx]
        data_dict.update(slide_id=slide_id)

        if self.data_load_mode == 0:
            slide, slide_label = self._load_slide_and_label(slide_id)
            data_dict.update(slide_label=slide_label)
        else:
            slide = self._load_slide(slide_id)
            data_dict.update(slide_label=self._load_slide_label(slide_id))

        data_dict.update(instance_centroid=slide.ndata['centroid'])
        data_dict.update(instance_feature=slide.ndata['feat'].squeeze(1))
        data_dict.update(instance_label=slide.ndata['label'])

        return data_dict


def Train_dataset(cfg, phase='train'):
    dataset = MixedData(cfg.Data, phase=phase)
    return dataset


def Test_dataset(cfg, phase='test'):
    dataset = MixedData(cfg.Data, phase=phase)
    return dataset


if __name__ == '__main__':
    cfg = read_yaml('configs/SICAPv2.yaml')
    Mydata = Train_dataset(cfg=cfg, phase='train')
    dataloader = DataLoader(Mydata)
    for i, data in (enumerate(dataloader)):
        pass
