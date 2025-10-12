# -*- coding:utf-8 -*-
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from causarray import prep_causarray_data, fit_gcate, LFC
import time
from causarray import estimate_r, plot_r

start_time = time.time()
# stim
adata_counts = sc.read_h5ad('data/SM2018_Tcells_counts.h5ad')
adata_counts_stim = adata_counts[adata_counts.obs.state=='Stim']
Y = pd.DataFrame(adata_counts_stim.X.A.copy(), columns=adata_counts_stim.var.index)
Y = Y.loc[:, ~(Y == 0).all()]
A = pd.get_dummies(adata_counts_stim.obs['condition'], columns=['condition'], drop_first=False).drop(columns=['control'])
Y, A, X, X_A = prep_causarray_data(Y, A)
Y = Y.astype(float)
A = A.astype(float)
df_r = estimate_r(Y, X, A, np.arange(5,55,5))
df_r.to_csv('SM2018_Tcells/stim/perturbseq_r.csv', index=False)
df_r = pd.read_csv('SM2018_Tcells/stim/perturbseq_r.csv')
fig = plot_r(df_r)

r = 15
res_1, res_2 = fit_gcate(Y, X, A, r, verbose=True)
U = res_2['U']

# save
#with open("SM2018_Tcells/stim/causarray_res1.pkl", "wb") as f:
#    pickle.dump(res_1, f)
#with open("SM2018_Tcells/stim/causarray_res2.pkl", "wb") as f:
#    pickle.dump(res_2, f)

offsets = np.log(res_2['kwargs_glm']['size_factor']) # use the precomputed size factors
df_res, estimation = LFC(Y, np.c_[X, U], A, np.c_[X_A, U], offset=offsets, verbose=True)
df_res.to_csv('SM2018_Tcells/stim/causarray_df_res.csv')

# unstim
adata_counts = sc.read_h5ad('data/SM2018_Tcells_counts.h5ad')
adata_counts_unstim = adata_counts[adata_counts.obs.state=='NoStim']
Y = pd.DataFrame(adata_counts_unstim.X.A.copy(), columns=adata_counts_unstim.var.index)
Y = Y.loc[:, ~(Y == 0).all()]
A = pd.get_dummies(adata_counts_unstim.obs['condition'], columns=['condition'], drop_first=False).drop(columns=['control'])
Y, A, X, X_A = prep_causarray_data(Y, A)
Y = Y.astype(float)
A = A.astype(float)
df_r = estimate_r(Y, X, A, np.arange(5,55,5))
df_r.to_csv('SM2018_Tcells/unstim/perturbseq_r.csv', index=False)
df_r = pd.read_csv('SM2018_Tcells/unstim/perturbseq_r.csv')
fig = plot_r(df_r)

r = 30
res_1, res_2 = fit_gcate(Y, X, A, r, verbose=True)
U = res_2['U']

# save
#with open("SM2018_Tcells/unstim/causarray_res1.pkl", "wb") as f:
#    pickle.dump(res_1, f)
#with open("SM2018_Tcells/unstim/causarray_res2.pkl", "wb") as f:
#    pickle.dump(res_2, f)

offsets = np.log(res_2['kwargs_glm']['size_factor']) # use the precomputed size factors
df_res, estimation = LFC(Y, np.c_[X, U], A, np.c_[X_A, U], offset=offsets, verbose=True)
df_res.to_csv('SM2018_Tcells/unstim/causarray_df_res.csv')
    
end_time = time.time()
print("Running time: {:.2f} seconds".format(end_time - start_time))
