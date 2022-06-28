
__author__ = "Hao Bian"
import inspect
import importlib  # In order to dynamically import the library
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class DataInterface(pl.LightningDataModule):

    def __init__(self, cfg, **kwargs):
        """[summary]
        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """
        super().__init__()
        self.cfg = cfg
        self.train_batch_size = self.cfg.Data.train_dataset.batch_size
        self.train_num_workers = self.cfg.Data.train_dataset.num_workers
        self.test_batch_size = self.cfg.Data.test_dataset.batch_size
        self.test_num_workers = self.cfg.Data.test_dataset.num_workers
        # self.dataset_name = self.cfg.Data.dataset_name
        self.kwargs = kwargs
        self.load_data_module()
        self.configure_transform()

    def prepare_data(self):
        # 1. how to download
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(
                phase='train', cfg=self.cfg)
            self.val_dataset = self.instancialize(
                phase='valid', cfg=self.cfg)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = self.instancialize(
                phase='test', cfg=self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_batch_size = self.train_batch_size

        return DataLoader(self.val_dataset, batch_size=val_batch_size, num_workers=self.train_num_workers, shuffle=True, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False, pin_memory=True)

    def configure_transform(self):
        self.transform = None

    def load_data_module(self):
        """  
        py文件命名为xx_data, 导入xx_data的XxData, 保存在self.data_module中
        """

        if not self.cfg.Data.train_dataset_name:
            train_dataset_name = 'Train_dataset'
        else:
            train_dataset_name = self.cfg.Data.train_dataset_name

        if not self.cfg.Data.test_dataset_name:
            test_dataset_name = 'Test_dataset'
        else:
            test_dataset_name = self.cfg.Data.test_dataset_name
        try:
            self.traindata_module = getattr(importlib.import_module(
                self.cfg.Package_name.data_name), train_dataset_name)
        except:
            raise ValueError(
                f'cannot from {self.cfg.Package_name.data_name} import {train_dataset_name}')
        try:
            self.testdata_module = getattr(importlib.import_module(
                self.cfg.Package_name.data_name), test_dataset_name)
        except:
            raise ValueError(
                f'cannot from {self.cfg.Package_name.data_name} import {test_dataset_name}')

    def instancialize(self, **args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        if args['phase'] == 'train' or args['phase'] == 'valid':
            self.data_module = self.traindata_module
        else:
            self.data_module = self.testdata_module
        return self.data_module(**args)
