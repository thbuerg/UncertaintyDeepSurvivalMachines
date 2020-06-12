import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from scipy import stats
from torchsupport.data.io import to_device, netread
from .plotting import set_style

colors, color_list = set_style()

# encode the train set:
def encode_dataset(model, ds, device='cpu', vae=False):
    model.eval()
    encoded = []

    #for sample in tqdm(ds):
    for sample in ds:
            sample = to_device(sample, device)
        if vae:
            encoded.append(model(sample.unsqueeze(0))[0].detach().cpu())
        else:
            encoded.append(model(sample.unsqueeze(0)).detach().cpu())

    features = np.concatenate(encoded, axis=0)
    var_df = pd.DataFrame([], columns=[], index=['lt_%d' % d for d in range(1, features.shape[1] + 1)])
    h5ad = ad.AnnData(X=features, var=var_df, obs=ds.meta)
    return h5ad


def calculate_umap(h5ad):
    sc.pp.neighbors(h5ad, use_rep='X')
    sc.tl.umap(h5ad)
    return h5ad


def plot_obs_umaps(h5ad, obs_cols, clip=True, normalize=True):
    """
    TODO: revise the normalization! -> use vmin and vmax from scpy
    :param h5ad:
    :param obs_cols:
    :param clip:
    :param normalize:
    :return:
    """

    if clip:
        stdsc = StandardScaler().fit(h5ad.obs[obs_cols].values)

        h5ad.obs[obs_cols] = h5ad.obs[obs_cols].mask(stdsc.transform(h5ad.obs[obs_cols].values) < -3)
        for c in obs_cols:
            h5ad.obs[c].fillna(-3 * h5ad.obs[c].std())
        h5ad.obs[obs_cols] = h5ad.obs[obs_cols].mask(stdsc.transform(h5ad.obs[obs_cols].values) > 3)
        for c in obs_cols:
            h5ad.obs[c].fillna(3 * h5ad.obs[c].std())

    if normalize:
        pete = PowerTransformer(method='yeo-johnson', standardize=True)
        h5ad.obs[obs_cols] = pete.fit_transform(h5ad.obs[obs_cols].values)

    figures = []
    # fig = plt.figure(figsize=(4*(len(obs_cols)//4), 4*2), dpi=150)
    for i, c in enumerate(obs_cols):
        # axs = fig.add_subplot(4, len(obs_cols)/4 + 1, i+1)
        sc.pl.umap(h5ad, color=c, return_fig=True, cmap='RdYlBu_r')
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        figures.append(fig)

    return figures