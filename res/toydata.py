# -*- coding:utf-8 -*-
"""
name:Toy data training
"""

import seaborn as sns
from CausalPerturb import causalperturb as cp
from CausalPerturb import plotting
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import torch

# The toy dataset is simulated from the PBMC dataset, with 3k cells and one perturbation
# that increases IFN-gamma–related gene expression specifically in B cells.
def main():
    dataset_name = 'toydata'  # to store results
    data_path = 'data/toydata.h5ad'  # data path
    adata = sc.read_h5ad(data_path)  # read data

    # Initialize W using oNMF, ~ 15 minutes
    cp.onmf(data=adata.X.T, dataset_name=dataset_name, ncells=500, nfactors=list(range(5, 16)))


    # Tuning parameters
    cp.model_train(data_path=data_path, dataset_name=dataset_name, perturbation_key='condition', split_key=None,
                     max_epochs=300, verbose=True)

    # Read basal, factors and loading
    basal = sc.read_h5ad(os.path.join(dataset_name, 'train_res', 'model_index={}_basal.h5ad'.format(0)))  # basal state
    treated = sc.read_h5ad(
        os.path.join(dataset_name, 'train_res', 'model_index={}_treated.h5ad').format(0))  # outcome factor state
    gene_loading = np.load(
        os.path.join(dataset_name, 'train_res', 'model_index={}_gene_loading.npy').format(0))  # gene loading matrix
    gene_loading_df = pd.DataFrame(gene_loading, columns=adata.var_names)

    # IFN related genes
    IFN_genes = ['ISG15', 'ISG20', 'IFI6', 'IFIT3', 'IFIT1', 'MT2A', 'SAMD9', 'HERC5',
                 'EIF2AK2', 'NT5C3A', 'DDX58', 'CMPK2', 'GIMAP4', 'TRIM22', 'RBCK1',
                 'GIMAP5', 'SP100', 'PPM1K', 'SOCS1', 'PLAC8', 'C19orf66', 'PHF11',
                 'PARP9', 'OAS3', 'C5orf56', 'RTP4', 'TREX1', 'IFITM1', 'BBX', 'IFIT5',
                 'ANXA2R', 'GIMAP1', 'PNPT1', 'ODF2L', 'TMX1']

    # Plot the loading for these genes
    plt.figure(figsize=(8.5, 4))
    sns.heatmap(gene_loading_df[IFN_genes], xticklabels=True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Gene loading', fontsize=12)
    plt.xlabel('IFN-gamma genes', fontsize=12)
    plt.ylabel('Factors', fontsize=12)
    plt.savefig(os.path.join(dataset_name, 'gene_loading.png'), bbox_inches='tight', dpi=1000)
    plt.close()
    
    row_sums = np.sum(gene_loading_df[IFN_genes].values,axis=1)
    max_row_index = np.argmax(row_sums)
    print('Factor {} is highly related, so we study the perturbation effects on Factor {}.'.format(max_row_index,max_row_index) )
    
    # Run causal forest
    tau_factor_mean, tau_q_val_factor, sig_factor = \
        cp.CF_single_target_single_factor(target='perturbed', factor=max_row_index, basal=basal,
                                              treated=treated, pert_key='condition',
                                              n_estimators=500, min_samples_leaf=10,
                                              random_state=140, alpha=0.05)

    adata.obs['tau_factor_mean'] = tau_factor_mean # perturbation effects
    adata.obs['tau_q_val_factor'] = tau_q_val_factor # q value
    adata.obs['sig_factor'] = sig_factor # significant or not

    # Plot result
    with plt.rc_context({'figure.figsize': (8, 5), 'font.sans-serif': ['Arial']}):
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        sc.pl.pca(adata[adata.obs.control == 1], color='cell_type', title='cell type', legend_loc='on data', frameon=False,
                  ax=axs[0][0], show=False)
        sc.pl.pca(adata[adata.obs.control == 1], color='tau_factor_mean', title='pert effects', frameon=False,
                  legend_fontsize=9, legend_fontoutline=1, ax=axs[0][1], show=False)
        sc.pl.pca(adata[adata.obs.control == 1], color='tau_q_val_factor', title='q value', frameon=False, ax=axs[1][0],
                  show=False)
        sc.pl.pca(adata[adata.obs.control == 1], color='sig_factor', title='sig', frameon=False, ax=axs[1][1], show=False)
        plt.savefig(os.path.join(dataset_name, 'CF_res.png'), bbox_inches='tight', dpi=1000)
        plt.show()

    # The perturbation effect is significant only in B cells, as expected.

if __name__ == '__main__':
    main()
