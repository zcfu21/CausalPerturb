# -*- coding:utf-8 -*-

import scanpy as sc
import pickle

data_path = "data/SM2018_Tcells_processed.h5ad"
adata = sc.read_h5ad(data_path)
adata_unstim = adata[adata.obs.state == 'NoStim']
adata_stim = adata[adata.obs.state == 'Stim']
pert_list = list(adata.obs.condition.cat.categories)

# stim
t_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_stim[adata_stim.obs.condition.isin([pert, 'control'])]
    sc.tl.rank_genes_groups(adata_subset, groupby='condition', method='t-test')
    degs = sc.get.rank_genes_groups_df(adata_subset, group=pert,
                                       pval_cutoff=0.05)
    t_degs_df_list.append(degs)
    print('Finish: {}'.format(pert))
with open("SM2018_Tcells/stim/t_degs_df_list.pkl", "wb") as f:
    pickle.dump(t_degs_df_list, f)

wilcox_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_stim[adata_stim.obs.condition.isin([pert, 'control'])]
    sc.tl.rank_genes_groups(adata_subset, groupby='condition', method='wilcoxon')
    degs = sc.get.rank_genes_groups_df(adata_subset, group=pert,
                                       pval_cutoff=0.05)
    wilcox_degs_df_list.append(degs)
    print('Finish: {}'.format(pert))
with open("SM2018_Tcells/stim/wilcox_degs_df_list.pkl", "wb") as f:
    pickle.dump(wilcox_degs_df_list, f)

# unstim
t_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_unstim[adata_unstim.obs.condition.isin([pert, 'control'])]
    sc.tl.rank_genes_groups(adata_subset, groupby='condition', method='t-test')
    degs = sc.get.rank_genes_groups_df(adata_subset, group=pert,
                                       pval_cutoff=0.05)
    t_degs_df_list.append(degs)
    print('Finish: {}'.format(pert))
with open("SM2018_Tcells/unstim/t_degs_df_list.pkl", "wb") as f:
    pickle.dump(t_degs_df_list, f)

wilcox_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_unstim[adata_unstim.obs.condition.isin([pert, 'control'])]
    sc.tl.rank_genes_groups(adata_subset, groupby='condition', method='wilcoxon')
    degs = sc.get.rank_genes_groups_df(adata_subset, group=pert,
                                       pval_cutoff=0.05)
    wilcox_degs_df_list.append(degs)
    print('Finish: {}'.format(pert))

with open("SM2018_Tcells/unstim/wilcox_degs_df_list.pkl", "wb") as f:
    pickle.dump(wilcox_degs_df_list, f)
