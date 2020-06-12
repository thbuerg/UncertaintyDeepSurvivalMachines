import numpy as np
import anndata as ad
import pandas as pd
pd.options.mode.use_inf_as_na = True

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchsupport.data.io import to_device, netread
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pycox.evaluation.concordance import concordance_td
from pycox.evaluation.eval_surv import EvalSurv

from mechanics.datasets import MetabricDataset
from mechanics.training import UncertaintyDeepSurvivalMachineTraining
from mechanics.labeltransform import LabTransFlexible
from mechanics.modules import ResNetBlock1d, ResNetUpBlock, FlattenLayer, PrintLayer
from mechanics.utils import setup


# Networks
class MLP(nn.Module):
    def __init__(self, in_dim=32, out_dim=2):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 256),
            nn.Dropout(0.3),
            nn.SELU(True),
            nn.Linear(256, 256),
            nn.SELU(True),
            nn.Linear(256, self.out_dim),
            nn.SELU(True),
        )

    @staticmethod
    def _init_gaussian(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.2, std=0.001)

    def forward(self, input, *args):
        features = self.mlp(input)
        return features


def evaluate(logger, FLAGS, valid_ds, training):
    """
    Evaluate Training based on Ctd index as defined in pycox.

    evaluate the training run and the model using PyCox EvalSurv Class


    :param logger:
    :param FLAGS:
    :param valid_ds:
    :param training:
    :return:
    """
    # first get predictions over time interval:
    eval_times = np.arange(1, valid_ds.durations['duration'].max())

    S_tX, F_tX, f_tX, eval_times,\
            durations, events = training.predict_dataset(training.valid_ds, eval_times)

    # get idx of event eval_times
    eval_times = eval_times.cpu().numpy()
    durations = durations.cpu().numpy()
    events = events.cpu().numpy()
    S_tX = S_tX.cpu().numpy()
    F_tX = F_tX.cpu().numpy()
    f_tX = f_tX.cpu().numpy()

    # construc pandas.df of survival fn:
    surv_fn = pd.DataFrame(S_tX, columns=[str(s) for s in durations], index=eval_times)
    surv_fn.to_csv(os.path.join('/data/project/uk_bb/cvd/results', 'S_tX.csv'))
    # get eval surv
    evaluation = EvalSurv(surv=surv_fn, durations=durations, events=events,
                          censor_surv='km', censor_durations=None, steps='post')
    C = evaluation.concordance_td(method='adj_antolini')
    print(C)

    # construct other dfs and save:
    surv_fn = pd.DataFrame(F_tX, columns=[str(s) for s in durations], index=eval_times)
    surv_fn.to_csv(os.path.join('/data/project/uk_bb/cvd/results', 'F_tX.csv'))
    surv_fn = pd.DataFrame(f_tX, columns=[str(s) for s in durations], index=eval_times)
    surv_fn.to_csv(os.path.join('/data/project/uk_bb/cvd/results', 'f_tX.csv'))


def train(logger, FLAGS, datasets):
    net = MLP(in_dim=FLAGS.inputdim, out_dim=FLAGS.zdim)

    networks = {
        'net': net
    }

    training = UncertaintyDeepSurvivalMachineTraining(networks, datasets['train'], datasets['valid'],
                                           alpha=FLAGS.alpha,
                                           gamma=FLAGS.gamma,
                                           delta=FLAGS.delta,
                                           kdim=FLAGS.kdim,
                                           zdim=FLAGS.zdim,
                                           schedule=None,
                                max_epochs=FLAGS.maxepochs,
                                batch_size=FLAGS.batchsize,
                                accumulate=None,
                                device=FLAGS.device,
                                network_name=FLAGS.networkname,
                                distribution=FLAGS.distribution,
                                path_prefix=FLAGS.out_directory,
                                report_interval=FLAGS.reportinterval,
                                checkpoint_interval=FLAGS.checkpointinterval,
                                optimizer=torch.optim.Adam,
                                optimizer_kwargs={"lr": FLAGS.learningrate},  # 2e-4},
                                verbose=True,
                                valid_callback=lambda x: None)

    # TODO: add some logs here
    #

    # run training
    trained_models = training.train()

    return training, trained_models


def prepare_datasets(logger):
    """
    Read the data into mem. Works fine for the current amount of ECG data,
     in future we might have to change to lazy behavior.
    :param split_data_path:
    :param ecgpath:
    :return:
    """
    logger.info('Preparing datasets')

    labeltransformer = None
    datatransformer = StandardScaler(copy=True)

    ds_train = MetabricDataset(mode='train', labeltransform=labeltransformer, datatransform=datatransformer)
    ds_valid = MetabricDataset(mode='valid', labeltransform=ds_train.labeltransform,
                               datatransform=ds_train.datatransform)

    # clip:
    ds_valid.durations['duration'] = ds_valid.durations['duration'].apply(lambda x: x if x>0 else 0.0001)

    return {'train': ds_train, 'valid': ds_valid}


if __name__ == '__main__':
    FLAGS, logger = setup(running_script="train_model.py",
                          config='./config.json',
                          logging_name='train_model')

    datasets = prepare_datasets(logger)

    training, models = train(logger, FLAGS, datasets)

    # evaluate(logger, FLAGS, datasets['valid'], training)

    logger.info('Done.')
