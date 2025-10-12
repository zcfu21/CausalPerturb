# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from sklearn.linear_model import LinearRegression
import random
import time
from statsmodels.stats.multitest import multipletests

data_path = "data/SM2018_Tcells_processed.h5ad"
adata = sc.read_h5ad(data_path)
adata_unstim = adata[adata.obs.state == 'NoStim']
adata_stim = adata[adata.obs.state == 'Stim']

def permute(G, Y, Be, n_permut=500):
    pval_mtx = np.zeros_like(Be.values)
    G_values = G.values
    for i in range(n_permut):
        random.seed(42 + i)
        G_pert = G_values[random.sample(np.arange(0, G.shape[0]).tolist(), G.shape[0]), :]
        enet = sklearn.linear_model.ElasticNet(precompute=True, l1_ratio=0.5, alpha=0.0005, max_iter=10000)
        enet.fit(G_pert, Y)
        pert_Be = pd.DataFrame(enet.coef_, columns=G.columns)
        pval_mtx = pval_mtx + (abs(pert_Be.values) >= abs(Be.values)) * 1
        if (i + 1) % 100 == 0:
            print("Finish {}".format(i))
    return pval_mtx / n_permut


start_time = time.time()
# stim
Y = adata_stim.X.todense()
X = pd.get_dummies(adata_stim.obs.condition)
X = X.drop('control', axis=1)
enet = sklearn.linear_model.ElasticNet(precompute=True, l1_ratio=0.5, alpha=0.0005, max_iter=10000)
enet.fit(X, Y)
Be = pd.DataFrame(enet.coef_)
Be.columns = X.columns
Be.index = adata_stim.var_names.values
Be.to_csv('SM2018_Tcells/stim/mimosca_beta.csv')
MIMOSCA_pval_mtx = permute(X, Y, Be, n_permut=500)
pvals = MIMOSCA_pval_mtx.flatten()
_, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
MIMOSCA_qval_mtx = pvals_corrected.reshape(MIMOSCA_pval_mtx.shape)
np.save("SM2018_Tcells/stim/mimosca_qval.npy", MIMOSCA_qval_mtx)

# unstim
Y = adata_unstim.X.todense()
X = pd.get_dummies(adata_unstim.obs.condition)
X = X.drop('control', axis=1)
enet = sklearn.linear_model.ElasticNet(precompute=True, l1_ratio=0.5, alpha=0.0005, max_iter=10000)
enet.fit(X, Y)
Be = pd.DataFrame(enet.coef_)
Be.columns = X.columns
Be.index = adata_unstim.var_names.values
Be.to_csv('SM2018_Tcells/unstim/mimosca_beta.csv')
MIMOSCA_pval_mtx = permute(X, Y, Be, n_permut=500)
pvals = MIMOSCA_pval_mtx.flatten()
_, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
MIMOSCA_qval_mtx = pvals_corrected.reshape(MIMOSCA_pval_mtx.shape)
np.save("SM2018_Tcells/unstim/mimosca_qval.npy", MIMOSCA_qval_mtx)

end_time = time.time()
print("Running time: {:.2f} seconds".format(end_time - start_time))
