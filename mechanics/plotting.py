"""
Some nice and some even nicer plot functions.
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def set_style():
    style_folder = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))), 'resources', 'style')
    with open(os.path.join(style_folder, 'style.json'), "r") as style_file,\
            open(os.path.join(style_folder, 'colors.json'), "r") as color_file:
        plt.style.use(json.load(style_file))
        colors = json.load(color_file)
        color_list = [*colors.values()]
    return colors, color_list

colors, color_list = set_style()


# TODO: depract this !! and modify the other ecg plot fn to save on demand
def plot_ecg(reconstruction, swap=True, isnumpy=False, figsize=None):
    labels = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    if not isnumpy:
        reconstruction = reconstruction.cpu().detach().numpy()

    if swap:
        reconstruction = np.swapaxes(reconstruction, 1, 2)

    fig, ax = plt.subplots(8, figsize=figsize, sharex='all', sharey=False)
    #     fig.patch.set_visible(False)
    # ax.axis('off')
    for i, abl in enumerate(reconstruction[0]):
        ax[i].plot(abl, 'k')
        ax[i].axis('off')
        # ax[i].set_ylabel(labels[i])
    # fig.tight_layout()
    #     fig.savefig('out'+str(n)+source+'.jpg')
    #     fig.show()
    #     fig.clf()
    return fig


def plot_and_save_ecg(reconstruction, swap=True, isnumpy=False, figsize=None, name='', id=0):
    labels = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if not isnumpy:
        reconstruction = reconstruction.cpu().detach().numpy()

    if swap:
        reconstruction = np.swapaxes(reconstruction, 1, 2)

    fig, ax = plt.subplots(8, figsize=figsize, sharex='all', sharey=False)

    for i, abl in enumerate(reconstruction[0]):
        ax[i].plot(abl, 'k')
        ax[i].set_ylabel(labels[i])

    fig.tight_layout()
    fig.savefig('%s__%d.jpg' % (name, id),
                # dpi=300
                )
    plt.close()
    return fig

def simple_barplot(performance_dict, metric='metric', error_bars=False, annotate=False):
    """
    performance_dict holds instances to plot
    e.g. {"model1": metric value}
    :param data_dict:
    :return:
    """
    fig, ax = plt.subplots(1, figsize=(4, 4))
    models = performance_dict.keys()
    # vals = [np.mean(performance_dict[mo][metric]) for mo in models]
    # stds = [np.std(performance_dict[mo][metric]) for mo in models]
    vals = [np.mean(performance_dict[mo]) for mo in models]
    stds = [np.std(performance_dict[mo]) for mo in models]
    xs = [*range(len(models))]
    ax.bar(x=xs, height=vals,
           yerr=stds if error_bars else None,
           color=[color_list[0] for _ in range(len(models))])
    if annotate:
        for i in range(len(models)):
            weight = "bold" if vals[i] == max(vals) else "normal"
            ax.text(x=i,
                    y=0.95,
                    s="{:.2f}".format(vals[i]),
                    horizontalalignment='center',
                    fontweight=weight)
    ax.set_ylabel(metric)
    ax.set_xticks(xs)
    ax.set_xticklabels([*performance_dict.keys()], rotation=60, horizontalalignment="right")
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_title(metric)
    return fig
