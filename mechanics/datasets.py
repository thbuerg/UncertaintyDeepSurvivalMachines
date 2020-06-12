import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

import requests
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycox.preprocessing.label_transforms import LabTransCoxTime, LabTransDiscreteTime
from scipy.sparse import issparse
from collections import defaultdict


class SimpleSurvivalDataset(Dataset):
    """
    Reads a single h5ad file, returns data as well as selected columns from .obs.
    """
    def __init__(self, data,
                 time_col, event_col,
                 device,
                 label_transformer=None, fit=False):
        """
        Intialize the class.

        :param data:  Data in h5ad format
        :param time_col: column w/ durations
        :param event_col:  column w/ events -> should be binary
        :param device: `cpu` or `cuda`
        :param label_transformer:  a pycox labeltransformer or sklearn transformer
        :param fit: whether to fit the passed transformer for this instance (`True`) or just apply (`False`)
        """
        self.data = data

        # remove sparsity
        if issparse(self.data.X):
            self.data.X = self.data.X.A

        self.labtrans = label_transformer
        self.device = device

        labels = self.get_labels(self.data, time_col, event_col)
        if self.labtrans is not None:
            labels = self.transform_labels(labels, fit=fit)
        self.durations = labels[0]
        self.events = labels[1]

    def get_labels(self, data, time_col, event_col):
        # make sure data.obs ist numeric:
        data.obs = data.obs.apply(pd.to_numeric, errors='raise')
        return (data.obs[time_col].values, data.obs[event_col].values)

    def transform_labels(self, labels, fit):
        if fit:
            labels = self.labtrans.fit_transform(*labels)
        else:
            labels = self.labtrans.transform(*labels)
        return labels

    def __len__(self):
        return self.data.X.shape[0]

    def __getitem__(self, idx):
        eid_data = self.data.X[idx, :]
        eid_as_tensor = torch.from_numpy(eid_data)

        eid_duration = torch.Tensor([self.durations[idx]]).squeeze(-1).long()
        eid_event = torch.Tensor([self.events[idx]]).squeeze(-1).long()

        return eid_as_tensor, (eid_duration, eid_event)


class MetabricDataset(Dataset):
    """
    Dataset class for metabric
    """
    def __init__(self, mode='complete', labeltransform=None, datatransform=None):
        self._dataset_url = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"
        self.name = 'metabric'
        self._remotefilename = "metabric/metabric_IHC4_clinical_train_test.h5"
        self._feather = f"./{self.name}.h5"
        assert mode in ['train', 'valid', 'complete'], '`mode` should be one of [`train`, `valid`, `complete`]'
        self.mode = mode
        # TODO -> allow save?
        self._download()
        if labeltransform is not None:
            self.labeltransform = labeltransform
            self._transform_labels()
        else:
            self.labeltransform = None
        if datatransform is not None:
            self.datatransform = datatransform
            self._transform_data()
        else:
            self.datatransform = None

    def _to_df(self, datadict):
        df = pd.DataFrame(datadict['x'],
                     columns=['ft_%d' % i for i in range(datadict['x'].shape[1])])
        df['duration'] = datadict['t']
        df['event'] = datadict['e']
        return df

    def _transform_data(self):
        """
        Apply transformation to data.
        :return:
        """
        nonbinaries = [v for v in self.data.columns if not all(np.isin(pd.unique(self.data[v]), [0., 1.]))]
        if self.mode in ['train', 'complete']:
            self.datatransform.fit(self.data[nonbinaries].values)

        transformed_data = self.datatransform.transform(self.data[nonbinaries].values)
        self.data[nonbinaries] = transformed_data

    def _transform_labels(self):
        """
        transform labels by scaling between 0, 1.
        :return:
        """
        if self.mode in ['train', 'complete']:
            self.labeltransform.fit(self.durations['duration'].values, self.events['event'].values)

        transformed_durations, _ = self.labeltransform.transform(self.durations['duration'].values,
                                                                   self.events['event'].values)

        self.durations.loc[:, 'duration'] = transformed_durations




    def _download(self):
        url = self._dataset_url + self._remotefilename
        with requests.Session() as s:
            r = s.get(url)
            with open(self._feather, 'wb') as f:
                f.write(r.content)

        data = defaultdict(dict)
        with h5py.File(self._feather) as f:
            for ds in f:
                for array in f[ds]:
                    data[ds][array] = f[ds][array][:]

        os.unlink(self._feather)
        if self.mode == 'train':
            self.data = self._to_df(data['train'])
        elif self.mode == 'valid':
            self.data = self._to_df(data['test'])
        elif self.mode == 'complete':
            train = self._to_df(data['train'])
            test = self._to_df(data['test'])
            self.data = pd.concat([train, test]).reset_index(drop=True)

        self.data = self.data.reindex()
        self.durations = self.data[['duration']]
        self.events = self.data[['event']]
        self.data.drop(['duration', 'event'], axis=1, inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data.loc[idx].values)
        duration = torch.Tensor(self.durations.loc[idx].values)
        event = torch.Tensor(self.events.loc[idx].values)
        return data, (duration, event)