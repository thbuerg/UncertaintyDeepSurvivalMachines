"""
Exposed versions of some TS losses, also custom survival modelling stuff.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def label_ce(prediction, target):
    """Requires logits!"""
    result = F.cross_entropy(
      prediction, target,
      reduction='sum'
    ) / target.size(0)
    return result


def label_bce(prediction, target):
    result = F.binary_cross_entropy_with_logits(
      prediction, target,
      reduction='sum'
    ) / target.size(0)
    return result


def label_mse(prediction, target):
    result = F.mse_loss(
      prediction, target,
      reduction='sum'
    ) / target.size(0)
    return result


def cox_ph_loss(logh, durations, events, eps=1e-7):
    """
    Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
    This approximation is valid for data w/ low percentage of ties.

    Credit to Haavard Kamme/PyCox

    :param logh:
    :param durations:
    :param events:
    :param eps:
    :return:
    """
    # sort:
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    logh = logh[idx]

    # calculate loss:
    events = events.view(-1)
    logh = logh.view(-1)
    gamma = logh.max()
    log_cumsum_h = logh.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    pll = - logh.sub(log_cumsum_h).mul(events).sum().div(events.sum())

    return pll


