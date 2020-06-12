"""
Define custom training schemes based on the Torschsupport Training API

mjendrusch/torchsupport
"""
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsupport.modules.losses.vae as vl
from math import pi
from torchsupport.training.training import Training
from torchsupport.data.collate import DataLoader
from torchsupport.training.state import (
  TrainingState, NetNameListState, NetState
)
from torchsupport.data.io import netwrite, to_device
from tensorboardX import SummaryWriter

from pycox.evaluation.concordance import concordance_td
from sksurv.metrics import concordance_index_ipcw, _estimate_concordance_index


from torch.distributions import Normal, LogNormal, Weibull, transform_to
from scipy.stats import lognorm, weibull_min
from collections import OrderedDict
import pandas as pd

from .losses import *
from .plotting import *


class SupervisedTraining(Training):
    """Standard supervised training process.
    Args:
    net (Module): a trainable network module.
    train_data (DataLoader): a :class:`DataLoader` returning the training
                              data set.
    validate_data (DataLoader): a :class:`DataLoader` return ing the
                                 validation data set.
    optimizer (Optimizer): an optimizer for the network. Defaults to ADAM.
    schedule (Schedule): a learning rate schedule. Defaults to decay when
                          stagnated.
    max_epochs (int): the maximum number of epochs to train.
    device (str): the device to run on.
    checkpoint_path (str): the path to save network checkpoints.
    """
    checkpoint_parameters = Training.checkpoint_parameters + [
        TrainingState(),
        NetNameListState("network_names"),
        NetState("optimizer")
    ]

    def __init__(self, networks, train_data, valid_data,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 schedule=None,
                 max_epochs=50,
                 batch_size=128,
                 accumulate=None,
                 device="cpu",
                 network_name="network",
                 path_prefix=".",
                 report_interval=10,
                 checkpoint_interval=1000,
                 verbose=False,
                 valid_callback=lambda x: None):
        super(SupervisedTraining, self).__init__()
        self.batch_size = batch_size

        self.valid_callback = valid_callback
        self.network_name = network_name
        self.verbose = verbose
        self.device = device
        self.accumulate = accumulate

        self.train_ds = train_data
        self.valid_ds = valid_data

        self.train_writer = SummaryWriter(f"{path_prefix}/{network_name}/train")
        self.valid_writer = SummaryWriter(f"{path_prefix}/{network_name}/valid")
        # self.meta_writer = SummaryWriter(f"{path_prefix}/{network_name}/meta")

        self.max_epochs = max_epochs
        self.checkpoint_path = f"{path_prefix}/{network_name}/{network_name}-checkpoint"
        self.report_interval = report_interval
        self.checkpoint_interval = checkpoint_interval

        self.step_id = 0
        self.epoch_id = 0
        self.best = None

        self.validation_losses = []  # needed for the schedule
        self.current_losses = {}

        netlist = []
        self.network_names = []
        for network in networks:
            self.network_names.append(network)
            network_object = networks[network].to(self.device)
            setattr(self, network, network_object)
            netlist.extend(list(network_object.parameters()))

        self.current_losses = {}
        self.network_name = network_name

        self.epoch_id = 0
        self.step_id = 0

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 5e-4}

        self.optimizer = optimizer(
            netlist,
            **optimizer_kwargs
        )

        if schedule is None:
            self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        else:
            self.schedule = schedule

    def run_networks(self, data):
        inputs, labels = data
        #         if not isinstance(inputs, (list, tuple)):
        #             inputs = [inputs]
        predictions, *args = self.net(inputs)
        #         if not isinstance(predictions, (list, tuple)):
        #             predictions = [predictions]
        #         return [combined for combined in zip(predictions, labels)]
        return predictions, labels

    def loss(self, predictions, targets):
        # predictions = predictions[:, -1].unsqueeze(-1) # class 1
        ce = label_ce(predictions, targets)
        loss_val = ce

        self.current_losses["cross-entropy"] = float(ce)
        self.current_losses["total"] = float(loss_val)

        return loss_val

    def valid_loss(self, inputs):
        loss_val = self.loss(inputs)
        return loss_val

    def preprocess(self, data):
        """Takes and partitions input data into VAE data and args."""
        return data

    def step(self, data):
        if self.accumulate is None:
            self.optimizer.zero_grad()

        # data = self.preprocess(data
        data = to_device(data, self.device)
        args = self.run_networks(data)

        loss_val = self.loss(*args)
        loss_val.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)

        if self.accumulate is None:
            self.optimizer.step()

        elif self.step_id % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.each_step()
        return float(loss_val)

    def valid_step(self, data):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            if isinstance(data, (list, tuple)):
                data = [
                    to_device(point, self.device)
                    for point in data
                ]
            elif isinstance(data, dict):
                data = {
                    key: to_device(data[key], self.device)
                    for key in data
                }
            else:
                data = to_device(data, self.device)
            args = self.run_networks(data)

            loss_val = self.loss(*args)

        return float(loss_val)

    def validate(self, data):
        with torch.no_grad():
            self.net.eval()
            loss = self.valid_step(data)
            self.validation_losses.append(loss)
            self.each_validate()
        #             self.valid_callback(
        #                 self, to_device(data, "cpu"), to_device(outputs, "cpu")
        #               )
        self.net.train()

    def schedule_step(self):
        # TODO: when are we cleaning this list
        self.schedule.step(sum(self.validation_losses))

    def each_epoch(self):
        # write epoch spike
        self.train_writer.add_scalar("epoch", self.epoch_id, self.step_id)
        self.train_writer.add_scalar("epoch", 0, self.step_id + 1)

    def each_step(self):
        Training.each_step(self)
        for loss_name in self.current_losses:
            loss_float = self.current_losses[loss_name]
            self.train_writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
            self.train_writer.flush()

    def each_validate(self):
        for loss_name in self.current_losses:
            loss_float = self.current_losses[loss_name]
            self.valid_writer.add_scalar(f"{loss_name} loss", loss_float, self.step_id)
            self.valid_writer.flush()

    def train(self):
        """Trains a supervised model until the maximum number of epochs is reached."""
        for epoch_id in range(self.max_epochs):
            self.epoch_id = epoch_id
            self.train_data = None
            self.train_data = DataLoader(self.train_ds, batch_size=self.batch_size,
                                         num_workers=8, shuffle=True, drop_last=True)
            valid_iter = None
            if self.valid_ds is not None:
                self.valid_data = DataLoader(
                    self.valid_ds, batch_size=self.batch_size, num_workers=8,
                    shuffle=True, drop_last=True
                )
                valid_iter = iter(self.valid_data)
            for data in self.train_data:
                self.step(data)
                if self.step_id % self.checkpoint_interval == 0:
                    self.checkpoint()
                if self.valid_data is not None and self.step_id % self.report_interval == 0:
                    vdata = None
                    try:
                        vdata = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(self.valid_data)
                        vdata = next(valid_iter)
                    vdata = to_device(vdata, self.device)
                    self.validate(vdata)
                self.step_id += 1
                self.schedule_step()
            self.each_epoch()

        netlist = [
            getattr(self, name)
            for name in self.network_names
        ]
        return netlist

    def save_path(self):
        path = self.checkpoint_path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return os.path.join(path, 'save')

    def checkpoint(self):
        """Performs a checkpoint for all encoders and decoders."""
        for name in self.network_names:
            the_net = getattr(self, name)
            if isinstance(the_net, torch.nn.DataParallel):
                the_net = the_net.module
            netwrite(
                the_net,
                os.path.join(self.checkpoint_path,
                             f"{name}_epoch_{self.epoch_id}_step_{self.step_id}.torch"
                             )
            )
        self.each_checkpoint()


class SurvivalTraining(SupervisedTraining):
    """Standard Survival training process.
             Essentially DeepSurv Training

    """
    checkpoint_parameters = Training.checkpoint_parameters + [
        TrainingState(),
        NetNameListState("network_names"),
        NetState("optimizer")
    ]

    def __init__(self, networks, train_data, valid_data,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 schedule=None,
                 max_epochs=50,
                 batch_size=128,
                 accumulate=None,
                 device="cpu",
                 network_name="network",
                 path_prefix=".",
                 report_interval=10,
                 checkpoint_interval=1000,
                 verbose=False,
                 valid_callback=lambda x: None):
        super(SurvivalTraining, self).__init__(
            networks=networks,
            train_data=train_data,
            valid_data=valid_data,
            optimizer=torch.optim.Adam,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            max_epochs=max_epochs,
            batch_size=batch_size,
            accumulate=accumulate,
            device=device,
            network_name=network_name,
            path_prefix=path_prefix,
            report_interval=report_interval,
            checkpoint_interval=checkpoint_interval,
            verbose=verbose,
            valid_callback=valid_callback)

    def survival_loss(self, logh, durations, events):
        return cox_ph_loss(logh, durations, events)

    def loss(self, logh, durations, events):
        sl = self.survival_loss(logh, durations, events)
        loss_val = sl

        self.current_losses["cox-ph"] = float(sl)
        self.current_losses["total"] = float(loss_val)

        return loss_val

    def valid_loss(self, logh, durations, events):
        loss_val = self.loss(logh, durations, events)
        return loss_val

    def run_networks(self, data):
        inputs, labels = data

        logh, *args = self.net(inputs)

        return logh, labels, args

    def step(self, data):
        if self.accumulate is None:
            self.optimizer.zero_grad()

        data = to_device(data, self.device)

        args = self.run_networks(data)

        loss_val = self.loss(*args)
        loss_val.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)

        if self.accumulate is None:
            self.optimizer.step()

        elif self.step_id % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.each_step()
        return float(loss_val)

    def valid_step(self, data):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            if isinstance(data, (list, tuple)):
                data = [
                    to_device(point, self.device)
                    for point in data
                ]
            elif isinstance(data, dict):
                data = {
                    key: to_device(data[key], self.device)
                    for key in data
                }
            else:
                data = to_device(data, self.device)
            logh, labels, args = self.run_networks(data)

            loss_val = self.loss(logh, *labels)

        return float(loss_val)

    def validate(self, data):
        with torch.no_grad():
            self.net.eval()
            loss = self.valid_step(data)
            self.validation_losses.append(loss)
            self.each_validate()
        #             self.valid_callback(
        #                 self, to_device(data, "cpu"), to_device(outputs, "cpu")
        #               )
        self.net.train()

    def each_epoch(self):
        # write epoch spike
        self.train_writer.add_scalar("epoch", self.epoch_id, self.step_id)
        self.train_writer.add_scalar("epoch", 0, self.step_id + 1)

        if (self.epoch_id + 1) == self.max_epochs:
            # calculate C-Index
            bins = [.25, .5, .75, .99]

            Cs = self.calculate_ctd(train_ds=self.train_ds, valid_ds=self.valid_ds, quantile_bins=bins)

            for k, v in zip(bins, Cs):
                print((k, v))
                self.valid_writer.add_scalar("C_%s" % k, v, self.step_id)
            self.valid_writer.flush()

    def calculate_ctd(self, train_ds, valid_ds, quantile_bins=[.25, .5, .75, 1.]):
        """
        Calculate the Ctd on the quartiles of the valid set
        :return:
        """
        surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
        surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)

        # move to structured arrays:
        struc_surv_train = np.array([(e, d) for e, d in zip(surv_train[0], surv_train[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])
        struc_surv_valid = np.array([(e, d) for e, d in zip(surv_valid[0], surv_valid[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])

        self.net.eval()

        c_tds = []
        loader = DataLoader(valid_ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)
        for tau in [np.quantile(surv_valid[1, surv_valid[0]>0], q) for q in quantile_bins]:
            F_ts = []
            tau_ = to_device(torch.Tensor([tau]), self.device)
            # with torch.no_grad():
            for data, (duration, event) in loader:
                data = to_device(data, self.device)

                preds = self.predict_sample(data, tau_)
                F_t = preds[1]

                F_ts.append(F_t)

            F_ts = torch.cat(F_ts, axis=0).squeeze(-1).detach().cpu().numpy()
            C = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                       F_ts,
                                       tau=tau, tied_tol=1e-8)
            c_tds.append(C[0])

        self.net.train()

        return c_tds

    def predict_sample(self, data, duration):
        """
        Predict the survival function for a sample at a given timepoint.
        :param data:
        :param duration:
        :return:
        """
        raise NotImplementedError("Implement accding to network.")
        S_t = None
        F_t = None
        f_t = None
        return f_t, F_t, S_t

    def predict_dataset(self, ds, times):
        """
        Predict the survival function for each sample in the dataset at all durations in the dataset.

        Returns a pandas DataFrame where the rows are timepoints and the columns are the samples. Values are S(t|X)
        :param ds:
        :param times: a np.Array holding the times for which to calculate the risk.
        :return:
        """
        # durations should be a np.array with monotonous increase stating the values in which to calculate the C^td
        times = to_device(torch.Tensor(times), self.device)

        # collect event times:
        # we assume that sample is (data, (duration, event))
        events = []
        durations = []
        for i in tqdm(range(len(ds))):
            s, (d, e) = ds[i]
            events.append(e)
            durations.append(d)
        # get a loader:
        loader = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

        events = torch.cat(events, dim=0)
        durations = torch.cat(durations, dim=0)
        durations, sort_idxs = torch.sort(durations, dim=0, descending=False)

        events = to_device(events, self.device)
        durations = to_device(durations, self.device)

        # predict each sample at each duration:
        self.net.eval()
        with torch.no_grad():
            S_tX = []
            F_tX = []
            f_tX = []
            for sample in tqdm(loader):
                sample, (_,_) = sample
                S_sample = []
                F_sample = []
                f_sample = []
                sample = to_device(sample, self.device)
                for t in times:
                    preds = self.predict_sample(sample, t) # to fit the U-DSM
                    S_t = preds[0]
                    F_t = preds[1]
                    f_t = preds[2]
                    S_sample.append(S_t)
                    F_sample.append(F_t)
                    f_sample.append(f_t)
                S_tX.append(torch.stack(S_sample, dim=0))
                F_tX.append(torch.stack(F_sample, dim=0))
                f_tX.append(torch.stack(f_sample, dim=0))
            S_tX = torch.cat(S_tX, dim=-1)
            F_tX = torch.cat(F_tX, dim=-1)
            f_tX = torch.cat(f_tX, dim=-1)

        self.net.train()

        S_tX = S_tX[:, sort_idxs]
        F_tX = F_tX[:, sort_idxs]
        events = events[sort_idxs]

        return S_tX, F_tX, f_tX, times, durations, events


class DeepSurvivalMachineTraining(SurvivalTraining):
    """
    Implements the training of DeepSurvivalMachine:

    """
    def __init__(self, networks, train_data, valid_data,
                 kdim=8,
                 zdim=64,
                 alpha=0.75,
                 gamma=1e-8,
                 distribution='lognormal',
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 schedule=None,
                 max_epochs=50,
                 batch_size=128,
                 accumulate=None,
                 device="cpu",
                 network_name="network",
                 path_prefix=".",
                 report_interval=10,
                 checkpoint_interval=1000,
                 verbose=False,
                 valid_callback=lambda x: None):
        super().__init__(
            networks=networks,
            train_data=train_data,
            valid_data=valid_data,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            max_epochs=max_epochs,
            batch_size=batch_size,
            accumulate=accumulate,
            device=device,
            network_name=network_name,
            path_prefix=path_prefix,
            report_interval=report_interval,
            checkpoint_interval=checkpoint_interval,
            verbose=verbose,
            valid_callback=valid_callback)
        self.kdim = kdim
        self.zdim = zdim
        self.alpha = alpha
        self.gamma = gamma # equals lambda from the original paper
        self.kdim = kdim

        assert distribution in ['lognormal', 'weibull'], 'Currently only `lognormal` & `weibull` available.'
        self.distribution = distribution


        netlist = []
        self.network_names = []
        for network in networks:
            self.network_names.append(network)
            network_object = networks[network].to(self.device)
            setattr(self, network, network_object)
            netlist.extend(list(network_object.parameters()))

        self.zeros = to_device(torch.zeros((self.batch_size, 1)), self.device)

        # draw the log eta_0, log beta_0 from fit:
        eta_est, beta_est = self.estimate_initial_params()

        self.etas0 = to_device(torch.empty(self.kdim).normal_(mean=eta_est, std=0.005), self.device)
        self.betas0 = to_device(torch.empty(self.kdim).normal_(mean=beta_est, std=0.005), self.device)
        self.etas0.requires_grad_(True)
        self.betas0.requires_grad_(True)

        # get the actual parameters:
        self.etas = nn.Sequential(nn.Linear(self.zdim, self.kdim, bias=True),
                                  nn.Tanh() if self.distribution == 'lognormal' else nn.SELU(),
                                  ).to(self.device)
        self.betas = nn.Sequential(nn.Linear(self.zdim, self.kdim, bias=True),
                                   nn.Tanh() if self.distribution == 'lognormal' else nn.SELU(),
                                   ).to(self.device)
        self.ks = nn.Linear(self.zdim, self.kdim, bias=False).to(self.device)

        # register w/ optmizer:
        netlist.extend([self.etas0, self.betas0])
        for ps in [self.etas, self.betas, self.ks]:
            netlist.extend(list(ps.parameters()))

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 5e-4}

        self.optimizer = optimizer(
            netlist,
            **optimizer_kwargs
        )

    def estimate_initial_params(self):
        """
        Initial parameters eta, beta are crucial for convergence. We estimate them by fitting the dist to 10 batches
        and averaging the params

        :return:
        """
        uncensored_durations = []
        durations = []
        for i in range(len(self.train_ds)):
            _, (duration, event) = self.train_ds[i]
            durations.append(duration)
            if event > 0:
                uncensored_durations.append(duration)
        uncensored_durations = torch.cat(uncensored_durations).numpy()

        if self.distribution == 'lognormal':
            mean = np.log(uncensored_durations).mean()
            std = np.log(uncensored_durations).std()
            eta = mean
            beta = np.log(std)
        elif self.distribution == 'weibull':
            shape, _, scale = weibull_min.fit(uncensored_durations, floc=0)
            eta = np.log(shape)
            beta = -np.log(scale)
        else:
            raise NotImplementedError('Currently only `lognormal` & `weibull` available.')

        print((eta, beta))
        return eta, beta

    def step(self, data):
        if self.accumulate is None:
            self.optimizer.zero_grad()

        data = to_device(data, self.device)
        data, (durations, events) = data

        args = self.run_networks(data, durations, events)

        loss_val = self.loss(durations, events, *args)
        loss_val.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)

        if self.accumulate is None:
            self.optimizer.step()

        elif self.step_id % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.each_step()

        return float(loss_val)

    def valid_step(self, data):
        """Performs a single step of VAE validation.
        Args:
          data: data points used for validation."""
        with torch.no_grad():
            if isinstance(data, (list, tuple)):
                data = [
                    to_device(point, self.device)
                    for point in data
                ]
            elif isinstance(data, dict):
                data = {
                    key: to_device(data[key], self.device)
                    for key in data
                }
            else:
                data = to_device(data, self.device)

            data, (durations, events) = data
            args = self.run_networks(data, durations, events)
            loss_val = self.loss(durations, events, *args)

        return float(loss_val)

    def censored_loss(self, S_ts, ks, events, e=0):
        """
        NLL on log hazards.
        :param logh:
        :param durations:
        :param events:
        :return:
        """
        elbo = torch.log(S_ts).add(F.log_softmax(ks, dim=-1)).where(events == e, self.zeros).sum()
        return -elbo/ks.size(0)

    def uncensored_loss(self, logf_ts, ks, events, e=1):
        """
        We minimize the ELBO of log P(DATASET_uncensored)
        equalling the negative sum over all log hazards.
        :param logf_t:
        :param durations:
        :param events:
        :return:
        """
        elbo = logf_ts.add(F.log_softmax(ks, dim=-1)).where(events == e, self.zeros).sum()
        return -elbo/ks.size(0)

    def prior_loss(self, etas, betas):
        """
        Regularize the priors with MSE

        :param etas:
        :param betas:
        :return:
        """
        e_mse = F.mse_loss(target=self.etas0.repeat(self.batch_size, 1), input=etas, reduce=False).sum()
        b_mse = F.mse_loss(target=self.betas0.repeat(self.batch_size, 1), input=betas, reduce=False).sum()
        pl = e_mse + b_mse
        return pl/etas.shape[0]

    def sample_survival(self, etas, betas, durations):
        """Sample from the distribution"""

        if self.distribution == 'lognormal':
            try:
                distribution = LogNormal(etas, torch.exp(betas), validate_args=True)
            except ValueError:
                print(etas[:3])  # shape/k
                print(torch.exp(betas)[:3])  # scale/lambda
                raise ValueError()
            logf_t = distribution.log_prob(durations)
            F_t = 0.5 + 0.5 * torch.erf(torch.div(torch.log(durations) - etas, np.sqrt(2)*torch.exp(betas)))
            S_t = 0.5 - 0.5 * torch.erf(torch.div(torch.log(durations) - etas, np.sqrt(2)*torch.exp(betas)))
        elif self.distribution == 'weibull':
            try:
                distribution = Weibull(
                    torch.exp(-betas),  # scale/lambda
                    torch.exp(etas),     # shape/k
                    validate_args=True
                )
                logf_t = distribution.log_prob(durations)
            except ValueError:
                print(torch.exp(etas))  # shape/k
                print(torch.exp(-betas))  # scale/lambda
                raise ValueError()

            F_t = 1 - torch.pow(torch.exp(-durations * torch.exp(betas)), torch.exp(etas))
            S_t = torch.pow(torch.exp(-durations * torch.exp(betas)), torch.exp(etas))
        else:
            raise NotImplementedError('Currently only `lognormal` & `weibull` available.')

        return logf_t.detach(), F_t.detach(), S_t.detach()

    def loss(self, durations, events, logf_t, S_t, batch_etas, batch_betas, etas, betas, ks):
        """ Calculate total DSM loss."""

        # TODO if multiple events -> apply censoring here and do stuff in iterations.
        elbo_u = self.uncensored_loss(logf_t, ks, events, e=1)
        elbo_c = self.censored_loss(S_t, ks, events, e=0)
        pl = self.prior_loss(batch_etas, batch_betas)

        loss_val = elbo_u \
                   + self.alpha * elbo_c \
                   + self.gamma * pl

        self.current_losses["4 prior_loss"] = float(pl)
        self.current_losses["3 ELBO_censored"] = float(elbo_c)
        self.current_losses["2 ELBO_uncensored"] = float(elbo_u)
        self.current_losses["1 total"] = float(loss_val)

        return loss_val

    def valid_loss(self, *args):
        loss_val = self.loss(*args)
        return loss_val

    def run_networks(self, data, durations, events):
        features = self.net(data)

        etas = self.etas(features)
        betas = self.betas(features)
        ks = self.ks(features)/1000
        batch_etas = self.etas0.repeat(self.batch_size, 1) + etas
        batch_betas = self.betas0.repeat(self.batch_size, 1) + betas

        logf_t, F_t, S_t = self.sample_survival(batch_etas, batch_betas, durations)

        return logf_t, S_t, batch_etas, batch_betas, etas, betas, ks

    def predict_sample(self, data, duration):
        """
        Predict a single sample.
        :param data:
        :param duration:
        :param event:
        :return:
        """
        features = self.net(data)

        etas = self.etas(features)
        betas = self.betas(features)
        ks = self.ks(features)/1000

        etas = self.etas0.repeat(1, 1) + etas
        betas = self.betas0.repeat(1, 1) + betas

        lf_t, F_t, S_t = self.sample_survival(etas, betas, duration)

        lS_t = torch.log(S_t)
        lS_t = lS_t + F.log_softmax(ks, dim=-1)
        S_t = torch.exp(torch.logsumexp(lS_t, dim=-1))

        lF_t = torch.log(F_t)
        lF_t = lF_t + F.log_softmax(ks, dim=-1)
        F_t = torch.exp(torch.logsumexp(lF_t, dim=-1))

        lf_t = lf_t + F.log_softmax(ks, dim=-1)
        f_t = torch.exp(torch.logsumexp(lf_t, dim=-1))

        return f_t, F_t, S_t


class VariationalDeepSurvivalMachineTraining(DeepSurvivalMachineTraining):
    """
    Implements the training of DeepSurvivalMachine:

    """
    def __init__(self,
                 networks,
                 train_data,
                 valid_data,
                 kdim=8,
                 zdim=64,
                 alpha=0.75,
                 gamma=1e-8,
                 delta=1.,
                 distribution='lognormal',
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 schedule=None,
                 max_epochs=50,
                 batch_size=128,
                 accumulate=None,
                 device="cpu",
                 network_name="network",
                 path_prefix=".",
                 report_interval=10,
                 checkpoint_interval=1000,
                 verbose=False,
                 valid_callback=lambda x: None):
        super().__init__(networks, train_data, valid_data, kdim=kdim, zdim=zdim,
                     alpha=alpha, gamma=gamma, distribution=distribution, optimizer=optimizer,
                     optimizer_kwargs=optimizer_kwargs, schedule=schedule, max_epochs=max_epochs,
                     batch_size=batch_size, accumulate=accumulate, device=device, network_name=network_name,
                     path_prefix=path_prefix, report_interval=report_interval, checkpoint_interval=checkpoint_interval,
                     verbose=verbose, valid_callback=valid_callback)
        self.delta = delta

    def sample_latent(self, mean, logvar):
        """
        Sample z from the latent space.
        :return:
        """
        distribution = Normal(mean, torch.exp(0.5 * logvar))
        sample = distribution.rsample()
        return sample

    def divergence_loss(self, mean, logvar):
        return vl.normal_kl_loss(mean, logvar)

    def run_networks(self, data, durations, events):
        features, mu, logvar = self.net(data)

        sample = self.sample_latent(mu, logvar)

        etas = self.etas(sample)
        betas = self.betas(sample)
        ks = self.ks(sample)/1000
        batch_etas = self.etas0.repeat(self.batch_size, 1) + etas
        batch_betas = self.betas0.repeat(self.batch_size, 1) + betas
        logf_t, F_t, S_t = self.sample_survival(batch_etas, batch_betas, durations)

        return logf_t, S_t, batch_etas, batch_betas, ks, (mu, logvar)

    def loss(self, durations, events, logf_t, S_t, batch_etas, batch_betas, ks, divergence_params):
        """ Calculate total DSM loss."""

        # TODO if multiple events -> apply censoring here and do stuff in iterations.
        elbo_u = self.uncensored_loss(logf_t, ks, events, e=1)
        elbo_c = self.censored_loss(S_t, ks, events, e=0)
        pl = self.prior_loss(batch_etas, batch_betas)
        div = self.divergence_loss(*divergence_params)

        loss_val = elbo_u \
                   + self.alpha * elbo_c \
                   + self.gamma * pl \
                   + self.delta * div

        self.current_losses["5 KL_loss"] = float(div)
        self.current_losses["4 prior_loss"] = float(pl)
        self.current_losses["3 ELBO_censored"] = float(elbo_c)
        self.current_losses["2 ELBO_uncensored"] = float(elbo_u)
        self.current_losses["1 total"] = float(loss_val)

        return loss_val

    # TODO: edit here
    def predict_sample(self, data, duration):
        """
        Predict a single sample.
        :param data:
        :param duration:
        :param event:
        :return:

        #TODO:  ADD UNCERTAINTY HERE!
        """
        _, mu, logvar = self.net(data)

        etas = self.etas(mu)
        betas = self.betas(mu)
        ks = self.ks(mu)/1000

        etas = self.etas0.repeat(1, 1) + etas
        betas = self.betas0.repeat(1, 1) + betas

        lf_t, F_t, S_t = self.sample_survival(etas, betas, duration)

        lS_t = torch.log(S_t)
        lS_t = lS_t + F.log_softmax(ks, dim=-1)
        S_t = torch.exp(torch.logsumexp(lS_t, dim=-1))

        lF_t = torch.log(F_t)
        lF_t = lF_t + F.log_softmax(ks, dim=-1)
        F_t = torch.exp(torch.logsumexp(lF_t, dim=-1))

        lf_t = lf_t + F.log_softmax(ks, dim=-1)
        f_t = torch.exp(torch.logsumexp(lf_t, dim=-1))

        return f_t, F_t, S_t


class UncertaintyDeepSurvivalMachineTraining(DeepSurvivalMachineTraining):
    """
    Implements the training of DeepSurvivalMachine:

    """
    def __init__(self,
                 networks,
                 train_data,
                 valid_data,
                 kdim=8,
                 zdim=64,
                 alpha=0.75,
                 gamma=1e-8,
                 delta=1.,
                 distribution='lognormal',
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs=None,
                 schedule=None,
                 max_epochs=50,
                 batch_size=128,
                 accumulate=None,
                 device="cpu",
                 network_name="network",
                 path_prefix=".",
                 report_interval=10,
                 checkpoint_interval=1000,
                 verbose=False,
                 valid_callback=lambda x: None):
        super().__init__(networks, train_data, valid_data, kdim=kdim, zdim=zdim,
                         alpha=alpha, gamma=gamma, distribution=distribution, optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs, schedule=schedule, max_epochs=max_epochs,
                         batch_size=batch_size, accumulate=accumulate, device=device, network_name=network_name,
                         path_prefix=path_prefix, report_interval=report_interval, checkpoint_interval=checkpoint_interval,
                         verbose=verbose, valid_callback=valid_callback)

        self.delta = delta

        # draw the log eta_0, log beta_0 from fit:
        eta_est, beta_est = self.estimate_initial_params()

        self.etas0 = to_device(torch.Tensor([eta_est]), self.device)
        self.betas0 = to_device(torch.Tensor([beta_est]), self.device)

        # get the actual parameters:
        self.etas_mu = nn.Sequential(nn.Linear(self.zdim, self.kdim, bias=True),
                                     # nn.Tanh() if self.distribution == 'lognormal' else nn.SELU(),
                                     ).to(self.device)
        self.etas_std = nn.Linear(self.zdim, self.kdim, bias=True).to(self.device)
        self.betas_mu = nn.Sequential(nn.Linear(self.zdim, self.kdim, bias=True),
                                      # nn.Tanh() if self.distribution == 'lognormal' else nn.SELU(),
                                      ).to(self.device)
        self.betas_std = nn.Linear(self.zdim, self.kdim, bias=True).to(self.device)
        self.ks_mu = nn.Linear(self.zdim, self.kdim, bias=True).to(self.device)
        self.ks_std = nn.Linear(self.zdim, self.kdim, bias=True).to(self.device)

        # add to optimizer
        netlist = []
        self.network_names = []
        for network in networks:
            self.network_names.append(network)
            network_object = networks[network].to(self.device)
            setattr(self, network, network_object)
            netlist.extend(list(network_object.parameters()))

        # register w/ optmizer:
        netlist.extend([self.etas0, self.betas0])
        for ps in [self.etas_mu, self.etas_std,
                   self.betas_mu, self.betas_std,
                   self.ks_mu, self.ks_std,
                   ]:
            netlist.extend(list(ps.parameters()))

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 5e-4}

        self.optimizer = optimizer(
            netlist,
            **optimizer_kwargs
        )

    def sample_latent(self, mean, sigma):
        """
        Sample z from the latent space.
        :return:
        """
        distribution = Normal(mean, sigma)
        sample = distribution.rsample()
        return sample

    def run_networks(self, data, durations, events):
        features = self.net(data)

        etas_mu = self.etas_mu(features)
        etas_sigma = self.etas_std(features)
        etas = self.sample_latent(etas_mu, etas_sigma)
        # etas = F.tanh(etas) if self.distribution == 'lognormal' else F.selu(etas)
        # print('eta')
        # print(etas[:2])

        betas_mu = self.betas_mu(features)
        betas_sigma = self.betas_std(features)
        betas = self.sample_latent(betas_mu, betas_sigma)
        # print('beta')
        # print(betas[:2])
        # betas = F.tanh(betas) if self.distribution == 'lognormal' else F.selu(betas)

        ks_mu = self.ks_mu(features)
        ks = ks_mu/1000
        # pre_ks_sigma = self.ks_std(features)
        # pre_ks_sigma = pre_etas_sigma
        # ks_sigma = 0.1 + 0.9*F.softplus(pre_ks_sigma) #TODO: check if we want the same sigma here?
        # ks = self.sample_latent(ks_mu, ks_sigma)/1000 # TODO: temperature on ks_mu

        # batch_etas = self.etas0.repeat(self.batch_size, 1) + etas
        # batch_betas = self.betas0.repeat(self.batch_size, 1) + betas

        logf_t, F_t, S_t = self.sample_survival(etas, betas, durations)

        return logf_t, S_t, etas, betas, ks, (etas_mu, etas_sigma), (betas_mu, betas_sigma)

    def divergence_loss(self, mean, sigma, r_mean=0.0, r_sigma=1.0):
        """
        Calculate KL divergence to normal.
        :param mean:
        :param sigma:
        :return:
        """
        distribution = Normal(mean, sigma)
        reference = Normal(r_mean, r_sigma)
        result = torch.distributions.kl_divergence(distribution, reference)
        return result.sum()

    def prior_loss(self, etas_params, betas_params):
        """
        Regularize the priors with MSE

        :param etas:
        :param betas:
        :return:
        """
        etas = Normal(etas_params[0], etas_params[1]).rsample()
        betas = Normal(betas_params[0], betas_params[1]).rsample()
        e_mse = F.mse_loss(target=self.etas0.repeat(self.batch_size, self.kdim), input=etas, reduce=False).sum()
        b_mse = F.mse_loss(target=self.betas0.repeat(self.batch_size, self.kdim), input=betas, reduce=False).sum()
        pl = e_mse + b_mse
        # e_nll = Normal(etas_params[0], etas_params[1]).log_prob(self.etas0.repeat(self.batch_size, self.kdim))
        # b_nll = Normal(betas_params[0], betas_params[1]).log_prob(self.betas0.repeat(self.batch_size, self.kdim))
        # pl = e_nll + b_nll
        return pl/etas_params[0].size()[0]


    def loss(self, durations, events, logf_t, S_t, batch_etas, batch_betas, ks, etas_params, betas_params):
        """ Calculate total DSM loss."""

        # TODO if multiple events -> apply censoring here and do stuff in iterations.
        elbo_u = self.uncensored_loss(logf_t, ks, events, e=1)
        elbo_c = self.censored_loss(S_t, ks, events, e=0)
        pl = self.prior_loss(etas_params, betas_params)
        etas_div = self.divergence_loss(*etas_params)
        betas_div = self.divergence_loss(*betas_params)

        # loss_val = elbo_u \
        #            + self.alpha * elbo_c \
        #            + self.gamma * pl
                   # + self.delta * div
        loss_val = pl if self.epoch_id < 5 else elbo_u \
                                                 + self.alpha * elbo_c\
                                                 + self.gamma * pl

        self.current_losses["6 betas_KL_loss"] = float(betas_div)
        self.current_losses["5 etas_KL_loss"] = float(etas_div)
        self.current_losses["4 prior_loss"] = float(pl)
        self.current_losses["3 ELBO_censored"] = float(elbo_c)
        self.current_losses["2 ELBO_uncensored"] = float(elbo_u)
        self.current_losses["1 total"] = float(loss_val)

        return loss_val

    def propagate_uncertainty(self, duration, ks_mu, ks_sigma, etas_mu, etas_sigma, betas_mu, betas_sigma):
        """
        Propagate the uncertainty estimates on k, eta, beta using taylor expansion.

        we use the formula:
        z = f(x_1, ..., x_n)
        \Delta z = \sqrt{\sum_{N}^{i=1}{\frac{\delta f}{\delta x_i} \Delta x_i}}

        where f() is the weighted mixture of either the CDF, 1-CDF or pdf of the distribution

        for logNormal:
            eta = mean
            beta = np.log(std)

        for Weibull:
            eta = np.log(shape) => shape = exp(eta)
            beta = -np.log(scale) => scale = exp(-beta)

        :param ks:
        :param etas:
        :param betas:
        :return:
        """
        S_t_error_terms = []
        F_t_error_terms = []
        f_t_error_terms = []

        if self.distribution == 'weibull':
            pdf_mixture = weibull_pdf_mixture
            cdf_mixture = weibull_cdf_mixture
            invcdf_mixture = weibull_invcdf_mixture

        elif self.distribution == 'lognormal':
            pdf_mixture = lognormal_pdf_mixture
            cdf_mixture = lognormal_cdf_mixture
            invcdf_mixture = lognormal_invcdf_mixture

        d_ks, d_etas, d_betas, _ = torch.autograd.functional.jacobian(cdf_mixture,
                                                                      inputs=(ks_mu, etas_mu, betas_mu, duration),
                                                                      # create_graph=True,
                                                                      strict=True)
        F_t_error_terms = [d_ks.mul(ks_sigma), d_etas.mul(etas_sigma), d_betas.mul(betas_sigma)]


        d_ks, d_etas, d_betas, _ = torch.autograd.functional.jacobian(invcdf_mixture,
                                                                      inputs=(ks_mu, etas_mu, betas_mu, duration),
                                                                      # create_graph=True,
                                                                      strict=True)
        S_t_error_terms = [d_ks.mul(ks_sigma), d_etas.mul(etas_sigma), d_betas.mul(betas_sigma)]

        d_ks, d_etas, d_betas, _ = torch.autograd.functional.jacobian(pdf_mixture,
                                                                      inputs=(ks_mu, etas_mu, betas_mu, duration),
                                                                      # create_graph=True,
                                                                      strict=True)
        f_t_error_terms = [d_ks.mul(ks_sigma), d_etas.mul(etas_sigma), d_betas.mul(betas_sigma)]


        S_t_error = torch.sum(torch.cat([x.pow(2) for x in S_t_error_terms])).sqrt().detach()
        F_t_error = torch.sum(torch.cat([x.pow(2) for x in F_t_error_terms])).sqrt().detach()
        f_t_error = torch.sum(torch.cat([x.pow(2) for x in f_t_error_terms])).sqrt().detach()

        # print((S_t_error, S_t))
        # print((F_t_error, F_t))
        # print((f_t_error, f_t))
        # raise NotImplementedError()

        return f_t_error, F_t_error, S_t_error

    # TODO: edit here
    def predict_sample(self, data, duration):
        """
        Predict a single sample.
        :param data:
        :param duration:
        :param event:
        :return:

        #TODO:  ADD UNCERTAINTY HERE!
        """
        features = self.net(data)

        etas_mu = self.etas_mu(features)
        etas_sigma = self.etas_std(features)

        betas_mu = self.betas_mu(features)
        betas_sigma = self.betas_std(features)

        ks_mu = self.ks_mu(features)/1000
        ks_sigma = self.ks_std(features)

        lf_t, F_t, S_t = self.sample_survival(etas_mu, betas_mu, duration)
        lS_t = torch.log(S_t)
        lS_t = lS_t + F.log_softmax(ks_mu, dim=-1)
        S_t = torch.exp(torch.logsumexp(lS_t, dim=-1))

        lF_t = torch.log(F_t)
        lF_t = lF_t + F.log_softmax(ks_mu, dim=-1)
        F_t = torch.exp(torch.logsumexp(lF_t, dim=-1))

        lf_t = lf_t + F.log_softmax(ks_mu, dim=-1)
        f_t = torch.exp(torch.logsumexp(lf_t, dim=-1))

        df_t, dF_t, dS_t = self.propagate_uncertainty(duration,
                                                      ks_mu,
                                                      ks_sigma,
                                                      etas_mu,
                                                      etas_sigma,
                                                      betas_mu,
                                                      betas_sigma
                                                      )

        return f_t, F_t, S_t, df_t, dF_t, dS_t

    def collect_uncertainties(self, ds, times):
        """
        Collect the uncertainties for the UDSM
        # TODO, delete this once debugging is done.
        :return:
        """
        # durations should be a np.array with monotonous increase stating the values in which to calculate the C^td
        times = to_device(torch.Tensor(times), self.device)

        # collect event times:
        # we assume that sample is (data, (duration, event))
        events = []
        durations = []
        for i in tqdm(range(len(ds))):
            s, (d, e) = ds[i]
            events.append(e)
            durations.append(d)
        # get a loader:
        loader = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

        events = torch.cat(events, dim=0)
        durations = torch.cat(durations, dim=0)
        durations, sort_idxs = torch.sort(durations, dim=0, descending=False)

        events = to_device(events, self.device)
        durations = to_device(durations, self.device)

        # predict each sample at each duration:
        self.net.eval()
        # with torch.no_grad():
        S_tX = []
        F_tX = []
        f_tX = []
        dS_tX = []
        dF_tX = []
        df_tX = []
        for sample in tqdm(loader):
            sample, (_,_) = sample
            S_sample = []
            F_sample = []
            f_sample = []
            dS_sample = []
            dF_sample = []
            df_sample = []
            sample = to_device(sample, self.device)
            for t in times:
                f_t, F_t, S_t, df_t, dF_t, dS_t = self.predict_sample(sample, t) # to fit the U-DSM
                S_sample.append(S_t)
                F_sample.append(F_t)
                f_sample.append(f_t)
                dS_sample.append(dS_t)
                dF_sample.append(dF_t)
                df_sample.append(df_t)
            S_tX.append(torch.stack(S_sample, dim=0))
            F_tX.append(torch.stack(F_sample, dim=0))
            f_tX.append(torch.stack(f_sample, dim=0))
            dS_tX.append(torch.stack(dS_sample, dim=0))
            dF_tX.append(torch.stack(dF_sample, dim=0))
            df_tX.append(torch.stack(df_sample, dim=0))
        S_tX = torch.cat(S_tX, dim=-1)
        F_tX = torch.cat(F_tX, dim=-1)
        f_tX = torch.cat(f_tX, dim=-1)
        dS_tX = torch.cat(dS_tX, dim=-1)
        dF_tX = torch.cat(dF_tX, dim=-1)
        df_tX = torch.cat(df_tX, dim=-1)

        self.net.train()

        S_tX = S_tX[:, sort_idxs]
        F_tX = F_tX[:, sort_idxs]
        f_tX = f_tX[:, sort_idxs]
        dS_tX = dS_tX[:, sort_idxs]
        dF_tX = dF_tX[:, sort_idxs]
        df_tX = df_tX[:, sort_idxs]
        events = events[sort_idxs]

        # print(S_tX.size())
        # raise NotImplementedError()

        return S_tX, F_tX, f_tX, times, durations, events

def weibull_invcdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull CDF based on two variables.

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    cdf = (1 - (1 - torch.pow(torch.exp(-durations * torch.exp(betas)), torch.exp(etas)))).mul(F.softmax(ks, dim=-1))
    return torch.sum(cdf, dim=-1)

def weibull_cdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull CDF based on two variables.

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    cdf = (1 - torch.pow(torch.exp(-durations * torch.exp(betas)), torch.exp(etas))).mul(F.softmax(ks, dim=-1))
    return torch.sum(cdf, dim=-1)

def weibull_pdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull pdf based on two variables:

    pdf(k, \lambda) = \frac{k}{\lambda} * (\frac{x}{\lambda})^{k-1}\exp{(\frac{-x}{\lambda})^{k}}

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    pdf = torch.exp(etas).mul(torch.exp(betas)).mul((durations.mul(torch.exp(betas))).pow(torch.exp(etas)-1))\
            .mul(torch.exp(-(durations.mul(torch.exp(betas))).pow(torch.exp(etas))))
    pdf_mix = torch.sum(pdf.mul(F.softmax(ks, dim=-1)), dim=-1)
    return pdf_mix

def lognormal_invcdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull CDF based on two variables.

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    cdf = 0.5 - 0.5 * (torch.erf((torch.log(durations) - etas).div((2*torch.exp(betas)).pow(0.5)))).mul(F.softmax(ks, dim=-1))
    return torch.sum(cdf, dim=-1)

def lognormal_cdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull CDF based on two variables.

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    cdf = 0.5 + 0.5 * (torch.erf((torch.log(durations) - etas).div((2*torch.exp(betas)).pow(0.5)))).mul(F.softmax(ks, dim=-1))
    return torch.sum(cdf, dim=-1)

def lognormal_pdf_mixture(ks, etas, betas, durations):
    """
    Defines the weibull pdf based on two variables:

    pdf(k, \lambda) = \frac{k}{\lambda} * (\frac{x}{\lambda})^{k-1}\exp{(\frac{-x}{\lambda})^{k}}

    This function is differentiable by autograd.
    :param etas:
    :param betas:
    :return:
    """
    pdf = ((durations*torch.exp(betas)*((2*pi)**0.5)).pow(-1))*torch.exp(-((torch.log(durations)-etas).pow(2)).div((2*torch.exp(betas)).pow(2)))
    pdf_mix = torch.sum(pdf.mul(F.softmax(ks, dim=-1)), dim=-1)
    return pdf_mix

