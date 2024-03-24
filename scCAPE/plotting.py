# -*- coding:utf-8 -*-
"""
name:Plotting
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os
import seaborn as sns
import matplotlib.colors as colors


def plot_top_genes_loadings(gene_names, W, figsize=(8,6), save_path=None, save=True):
    """
    Plot the top genes loadings for each factor and select the genes to do GSEA.
    Parameters
    ----------
    gene_names: list
    Gene names
    W: array
    Non-negative orthogonal loading matrix
    Returns
    -------
    The indices of top genes of each factor, which will be used to do GSEA
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    k = W.shape[1]
    stds = np.array(W.std(axis=0)).flatten()  # get the factor standard deviations
    stdfactor = 2  # factor to select genes
    genes_idx = np.array([])
    genes_loadings = np.array([])
    genes_do_go = {}
    for i, s in enumerate(stds):  # for each factor i and its respective standard deviations
        a = np.where(np.array(W[:, i]).flatten() > stdfactor * s)[0]  # get indices of top genes
        sub = np.array(W[a, i]).flatten()  # grab loadings of top genes for factor i
        a = a[np.argsort(sub)[::-1]]  # sort subset by loadings
        genes_do_go[str(i)] = a
        genes_idx = np.concatenate([genes_idx, a])  # concatenate gene indices
        genes_loadings = np.concatenate([genes_loadings, np.sort(sub)[::-1]])
    genes_idx = genes_idx.astype(int)

    # make list unique
    s = set()
    l = []
    vals = []
    for i in range(len(genes_idx)):
        x = genes_idx[i]
        currval = genes_loadings[i]
        if x in s:
            # find the existing loading value and compare it to the new loading value
            origidx = np.argwhere(l == x)[0][0]
            origval = vals[origidx]
            if origval < currval:
                # remove the origval and gene name in the list and append the new val and gene name
                l.remove(x)
                vals.remove(origval)
                l.append(x)
                vals.append(currval)

        if x not in s:
            s.add(x)
            l.append(x)
            vals.append(currval)

    genes_idx = np.array(l)[::-1]
    keptgenes = [gene_names[i] for i in genes_idx]

    # define gene font sizes
    geneFS = 3  # (default)
    if len(genes_idx) > 200:
        geneFS = 2
    if len(genes_idx) > 400:
        geneFS = 1
    if len(genes_idx) > 600:
        geneFS = 0.5

    mtx = W[genes_idx, :]  # subset feature space with the top genes for the features

    with plt.rc_context({'figure.figsize': figsize, 'font.sans-serif': ['Arial']}):
        ar = 2.5 * k / len(genes_idx)  # define aspect ratio
        plt.imshow(mtx, cmap='magma', aspect=ar, interpolation='none')  # create heatmap
        plt.ylabel('Genes')
        plt.ylim(-0.5, len(genes_idx) - 0.5)
        plt.yticks(np.arange(len(genes_idx)), keptgenes, fontsize=geneFS)
        # plt.yticks(np.arange(len(genes_idx)), [''] * len(genes_idx))
        plt.xticks(np.arange(k), np.arange(k) + 1)
        plt.xlabel('Factor')
        plt.colorbar(pad=0.02)
        if save:
            plt.savefig(os.path.join(save_path, 'Gene_loadings.svg'), bbox_inches='tight', dpi=1000)
        plt.show()
    return genes_do_go


def plot_loss(history, save_path, model_index):
    """
    Plot the loss along with the training epochs.
    """
    loss_list = ["loss_reconstruction", "loss_adv_perts", "l2loss"]
    subset_keys = ["epoch"] + loss_list
    loss_df = pd.DataFrame(
        dict((k, history[k]) for k in subset_keys if k in history))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(14, 4))
    for i in range(3):
        ax[i].plot(
            loss_df["epoch"].values, loss_df[loss_list[i]].values
        )
        ax[i].set_title(loss_list[i], fontweight="bold")
        ax[i].set_xlabel("epoch")
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'model_index={}_training_loss.png'.format(model_index)), dpi=1000)
    plt.close()


def plot_metric(history, save_path, model_index, epoch_min=0):
    """
    Plot the reconstruction R2 along with training epochs.
    """
    header = ["mean_all", "mean_DEG"]
    metric_df = pd.DataFrame(columns=["epoch", "split"] + header)
    for split in ["training", "test"]:
        df_split = pd.DataFrame(np.array(history[split]), columns=header)
        df_split["split"] = split
        df_split["epoch"] = history["stats_epoch"]
        metric_df = pd.concat([metric_df, df_split])
    metric_df = metric_df.melt(id_vars=["epoch", "split"])
    col_dict = dict(zip(["training", "test"], ["#377eb8", "#4daf4a"]))
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10, 4))
    for i in range(2):
        sns.lineplot(
            data=metric_df[(metric_df["variable"] == header[i]) & (metric_df["epoch"] > epoch_min)],
            x="epoch",
            y="value",
            palette=col_dict,
            hue="split",
            ax=axs[i]
        )
        axs[i].set_title(header[i], fontweight="bold")
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'model_index={}_training_metric.png'.format(model_index)), dpi=1000)
    plt.close()


def plot_CATE(data, target, tau_factor_mean, tau_q_val_factor, sig_factor, pert_key='condition',
              figsize=(9, 6),save_path=None, save=False):
    """
    Plot the perturbation effects on UMAP.
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    data_subset = data[data.obs[pert_key].isin([target, 'control'])]
    data_subset.obs['CATE'] = tau_factor_mean
    data_subset.obs['qvalue'] = tau_q_val_factor
    data_subset.obs['sig'] = sig_factor
    fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    sc.pl.umap(data_subset, color=pert_key, ax=axs[0, 0], show=False)
    sc.pl.umap(data_subset, color='CATE', ax=axs[0, 1], show=False)
    sc.pl.umap(data_subset, color='qvalue', ax=axs[1, 0], show=False)
    sc.pl.umap(data_subset, color='sig', ax=axs[1, 1], show=False)
    if save:
        plt.savefig(os.path.join(save_path, 'CATE_target={}.png'.format(target)), bbox_inches='tight', dpi=1000)
    plt.show()


def dotplot_pert_factor(clust_mean_df, clust_sig_df, cluster_name, color_min=None, color_max=None,
                        figsize=(6, 4), save_path=None, save=False):
    """
    Plot the average perturbation effects and proportion of significant cells aggregated on cell subpopulation
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    num_factors = clust_mean_df.shape[1] - 2
    mean_df = clust_mean_df[clust_mean_df['cell_state'] == cluster_name]
    mean_df.index = mean_df['target']
    mean_df = mean_df.drop(columns=['cell_state', 'target'])

    sig_df = clust_sig_df[clust_sig_df['cell_state'] == cluster_name]
    sig_df.index = sig_df['target']
    sig_df = sig_df.drop(columns=['cell_state', 'target'])

    target_names = mean_df.index
    mean_df = mean_df.reset_index(drop=True)
    sig_df = sig_df.reset_index(drop=True)

    mean_melt = mean_df.stack().reset_index().rename(
        columns={'level_0': 'target', 'level_1': 'factor', 0: 'CATE'})
    sig_melt = sig_df.stack().reset_index().rename(
        columns={'level_0': 'target', 'level_1': 'factor', 0: 'sig'})

    drawing_df = mean_melt
    drawing_df['sig'] = sig_melt['sig']
    drawing_df['factor'] = num_factors - 1 - drawing_df['factor']

    with plt.rc_context({'figure.figsize': figsize, 'font.sans-serif': ['Arial']}):
        fig, ax = plt.subplots()
        vcenter = 0.0
        if color_min is not None:
            vmin = color_min
        else:
            vmin = drawing_df['CATE'].min()
        if color_max is not None:
            vmax = color_max
        else:
            vmax = drawing_df['CATE'].max()
        normalize = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        colormap = plt.cm.coolwarm
        g = sns.scatterplot(data=drawing_df, x='target', y='factor', c=drawing_df['CATE'], size='sig', cmap=colormap,
                            norm=normalize, sizes=(30, 220))
        h, l = g.get_legend_handles_labels()
        plt.legend(h[0:7], l[0:7], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10, frameon=False,
                   title='sig %', labelspacing=0.7)
        plt.yticks(np.arange(num_factors), (np.arange(num_factors) + 1)[::-1])
        plt.xticks(np.arange(len(target_names)), target_names, rotation=45, fontsize=9)
        plt.xlabel('Target')
        plt.ylabel('Factor')
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
        sm.set_array([])
        cax = fig.add_axes([ax.get_position().x1 + 0.05, ax.get_position().y0, 0.06, ax.get_position().height / 2.5])
        cbar = ax.figure.colorbar(sm, cax=cax)
        cbar.ax.set_title('   CATE', fontsize=10)
        if save:
            plt.savefig(os.path.join(save_path, 'dotplot_pert_factor_on_{}.svg'.format(cluster_name)),
                        bbox_inches='tight', dpi=1000)
        plt.show()


def plot_similarity(df, colorbar_ticks=[0,20,40],figsize=(5.5, 5.5), name=None ,save_path=None, save=True):
    """
    Plot the pairwise distance matrix between perturbations
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")
    kws = dict(cbar_kws=dict(ticks=colorbar_ticks, orientation='horizontal'), figsize=figsize)
    with plt.rc_context({'font.sans-serif': ['Arial']}):
        g = sns.clustermap(df, dendrogram_ratio=0.2, cmap="rocket", **kws)
        x0, _y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0, 0.9, g.ax_row_dendrogram.get_position().width, 0.02])
        g.ax_cbar.set_title('Distance', fontsize=10)
        g.ax_cbar.tick_params(axis='x', length=3)
        if save:
            plt.savefig(os.path.join(save_path, 'pert_similarity_plot_{}.svg'.format(name)),
                        bbox_inches='tight', dpi=1000)
        plt.show()


def triangulation_for_triheatmap(M, N):
    """
    Make triangles for the heatmap representing genetic interactions.
    """
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

def GI_triangle_plot(BF_all_res, DM_all_res, SY_all_res, epiA_all_res, epiB_all_res,
                     target_list, show_factors='All', show_targets='All', vmin_1=0.2, vmin_2=0.3,
                     figsize=(26, 7), save_path=None, save=False):
    """
    Plot the genetic interactions for specified targets and factors.
    """
    if save:
        if save_path is None:
            raise Exception("Please provide save_path to save the plot.")

    if show_factors == 'All':
        show_factors_idx = list(np.arange(DM_all_res.shape[1]))
        show_factors = list(np.arange(DM_all_res.shape[1]))
    else:
        show_factors_idx = show_factors

    if show_targets == 'All':
        show_targets_idx = list(np.arange(DM_all_res.shape[0]))
        show_targets = target_list
    else:
        show_targets_idx = [target_list.index(target) for target in show_factors]

    M, N = len(show_targets), len(show_factors)  # e.g. M columns, N rows
    blank_all_res = np.ones_like(DM_all_res) * vmin_1
    values = [(BF_all_res[show_targets_idx, :][:, show_factors_idx]).T,
              (DM_all_res[show_targets_idx, :][:, show_factors_idx]).T,
              (SY_all_res[show_targets_idx, :][:, show_factors_idx]).T,
              (blank_all_res[show_targets_idx, :][:, show_factors_idx]).T]

    triangul = triangulation_for_triheatmap(M, N)
    cmaps = ['Reds', 'Greens', 'Purples', 'Blues']
    norms = [plt.Normalize(0, 1) for _ in range(4)]

    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    triangles1 = [(i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1)) for j in range(N) for i in range(M)]
    triangles2 = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1)) for j in range(N) for i in
                  range(M)]
    triang1 = Triangulation(x, y, triangles1)
    triang2 = Triangulation(x, y, triangles2)

    with plt.rc_context({'figure.figsize': figsize, 'font.sans-serif': ['Arial'], 'font.size': 15}):
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.4, 1]})
        # BF, DM, SY
        imgs = [ax[0].tripcolor(t, np.ravel(val), cmap=cmap, ec='white', alpha=1, vmin=vmin_1)
                for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]
        ax[0].set_xticks([])
        ax[0].set_yticks(range(N), np.array(show_factors_idx) + 1)
        ax[0].set_ylabel('Factor')
        ax[0].invert_yaxis()
        ax[0].margins(x=0, y=0)

        # EpiA, EpiB
        imgs1 = ax[1].tripcolor(triang1, np.ravel((epiA_all_res[show_targets_idx, :][:, show_factors_idx]).T), cmap='Oranges',
                                ec='white', vmin=vmin_2)
        imgs2 = ax[1].tripcolor(triang2, np.ravel((epiB_all_res[show_targets_idx, :][:, show_factors_idx]).T), cmap='Blues',
                                ec='white', vmin=vmin_2)
        ax[1].set_xticks(range(M), show_targets, rotation=90)
        ax[1].set_yticks(range(N), np.array(show_factors_idx) + 1)
        ax[1].set_ylabel('Factor')
        ax[1].invert_yaxis()
        ax[1].margins(x=0, y=0)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, 'SYBFDM_epi_res.svg'), dpi=1000, bbox_inches="tight")
        plt.show()