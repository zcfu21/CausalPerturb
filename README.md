# CausalPerturb

### Causal Analysis of heterogeneous Perturbation Effects in single cell CRISPR screening data

<img align="center" src="./overview.png?raw=true" width=550 height=570>

### Introduction
In CausalPerturb, we formulated the perturbation analysis in single-cell CRISPR screening data as a problem of treatment effect estimation.

CausalPerturb outputs the estimation and inference of heterogeneous perturbation effects at single-cell resolution by
1. disentangling the perturbation effects from inherent cell-state variations based on an autoencoder framework with adversarial training;
2. growing causal forests for each perturbation and factor.

CausalPerturb enables us to
* perform perturbation analysis at single-cell resolution,
* quantify the similarities between perturbations and prioritize the perturbation targets at any subpopulation level, either in an overall or a functional factor-specific context,
* and infer genetic interactions in high-MOI datasets.

### Installation
CausalPerturb is based on `python` version 3.9, `torch` and `scanpy`. Install directly from pip with:

```python
  pip install CausalPerturb
```

### Input 
The input adata of CausalPerturb contains the normalized cell-by-gene matrix and the following metadata:
1. **'condition'**: The perturbation labels. **Note the label of control cells must be 'control'**. **When analyzing high-MOI datasets, the perturbation label must be 'gene_a+gene_b' for two-gene perturbations**;
2. **'control'** (Not necessary) : The dummy variable to show if the cell is control (1) or perturbed (0), which will be generated based on perturbation label ('condition') when performing adversarial training in CausalPerturb.
3. **'cell_type'** (Not necessary) : The cell states used in downstream analysis, which can be pre-defined or clustered using some unsupervised algorithms like 'leiden'. If not specified, CausalPerturb will perform leiden clustering using 'sc.tl.leiden(data)' when performing adversarial training;

### Basic Usage
```python
from CausalPerturb import causalperturb as cp
from CausalPerturb import plotting
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import pickle
import gseapy as gp
from fractions import Fraction
dataset_name = 'dataset_name' # to store results
data_path = './data.h5ad' # data path
adata = sc.read_h5ad(data_path) # read data
```
1. Initialize the weight matrix by performing orthogonal non-negative matrix factorization:<br>
``` python
cp.onmf(data=adata.X.T, dataset_name=dataset_name, ncells=2000, nfactors=list(range(5, 16)))
```
**The initialized loading matrix** ('W_ini.npy') will be stored in "./dataset_name/oNMF". 

2. Disentangle perturbation effects from inherent cell-state variations using adversarial training:<br>
``` python
cp.model_train(data_path=data_path, dataset_name=dataset_name, perturbation_key='condition', split_key=None,
                     max_epochs=300, verbose=True)
```
**The model file** ('stored_model.pt'), **basal state**('model_basal.h5ad'), **outcome factor** ('model_treated.h5ad') and **gene loading matrix** ('model_gene_loading.npy') will be stored in "./dataset_name/train_res". Currently only supports training on CPU.

3. Growing causal forests for each perturbation and factor:<br>
```python
basal=sc.read_h5ad(os.path.join(dataset_name,'train_res','model_index=0_basal.h5ad')) # basal state
treated=sc.read_h5ad(os.path.join(dataset_name,'train_res','model_index=0_treated.h5ad')) # outcome factor state
cp.CF_all_target_all_factor(dataset_name=dataset_name, basal=basal, treated=treated, return_df=True,
                                 pert_key='condition', verbose=True, alpha=0.05)
```
The function outputs three dictionaries in "./dataset_name/CausalForests": **'tau_factor_mean_all_dict.pkl'**, **'tau_q_val_factor_all_dict.pkl'** and **'sig_factor_all_dict.pkl'**. The keys of the dicts are perturbations, and the values are num_cells\*num_factors array which gives perturbation effect, qvalue and significance on each factor of each cell. Similar csv files of each perturbation will be stored if setting return_df=True.

4. Factor annotation:<br>
```python
gene_loading=np.load(os.path.join(dataset_name,'train_res','model_index=0_gene_loading.npy')) # gene loading matrix
gene_loading_df=pd.DataFrame(gene_loading,columns=adata.var_names)
# selecting high-loading genes of each factor
genes_do_go=plotting.plot_top_genes_loadings(gene_names=adata.var_names,W=gene_loading.T, figsize=(10,6), save_path=None, save=False)
# Gene set enrichment analysis
for factor in range(gene_loading.shape[0]):
    go_res = gp.enrichr(gene_list=list(adata.var_names[genes_do_go[str(factor)]]),
                    organism='Human',
                    gene_sets='GO_Biological_Process_2023')
    go_res_df=go_res.results[go_res.results['Adjusted P-value']<=0.05]
    go_res_df['Term']=[term.split(' (')[0] for term in go_res_df.Term.values]
    go_res_df['Overlap']=[float(Fraction(i)) for i in go_res_df.Overlap.values]
    go_res_df.to_csv(os.path.join(dataset_name,'train_res','GO_factor_{}.csv'.format(factor)))
    print('Finish: {}'.format(factor))
```
The GO enrichment results will be stored in "./dataset_name/train_res".

Check the basic usage with a toy data ('toydata.py') in folder 'res'. 

The results in the manuscript and some additional analysis are also provided in folder 'res'.

### Other available functions

*  Genetic interaction analysis: `cp.CF_GI_single_target_single_factor`
*  Combine p-values to infer genetic interactions at the whole transcriptome level: `cp.cal_ACAT_pvals`
*  Calculate the perturbation similarity on a specific factor: `cp.cal_factor_level_similarity`
*  Calculate the overall perturbation similarity: `cp.cal_overall_similarity`
*  Calculate the perturbation rank list on a specific factor: `cp.cal_factor_level_rank`
*  Calculate the overall perturbation rank list: `cp.cal_overall_rank`

There are also some functions to generate plots: import the module `from CausalPerturb import plotting`  and use the plotting functions. 

### Contact
Please contact [fzc21@mails.tsinghua.edu.cn] with questions.
