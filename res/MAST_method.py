# -*- coding:utf-8 -*-
import logging
import random
import os
import anndata2ri
import rpy2.rinterface_lib.callbacks
import scanpy as sc
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
import rpy2.rinterface_lib.callbacks as cb
import time

# The following codes should be run in jupyter notebook cells

sc.settings.verbosity = 0
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

pandas2ri.activate()
anndata2ri.activate()
os.environ['R_ENCODING'] = 'UTF-8'

#######
%load_ext rpy2.ipython
#######

#######
%%R
Sys.setlocale("LC_CTYPE", "C")
#######

def safe_consolewrite(x):
    try:
        print(x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
    except Exception:
        print("[Non-UTF8 output suppressed]")

def prep_anndata(adata_):
    def fix_dtypes(adata_):
        df = pd.DataFrame(adata_.X.A, index=adata_.obs_names, columns=adata_.var_names)
        df = df.join(adata_.obs)
        return sc.AnnData(df[adata_.var_names], obs=df.drop(columns=adata_.var_names))

    adata_ = fix_dtypes(adata_)
    # sc.pp.filter_genes(adata_, min_cells=3)
    return adata_

cb.consolewrite_print = safe_consolewrite
cb.consolewrite_warnerror = safe_consolewrite

#######
# load MAST
%%R
library(MAST)
"MAST" %in% rownames(installed.packages())
if (require(MAST, quietly = TRUE)) {
  print("MAST loaded successfully!")
} else {
  print("MAST failed to load.")
}
exists("SceToSingleCellAssay", where="package:MAST")
#######

#######
%%R
find_de_MAST_RE <- function(adata_){
    # create a MAST object
    sca <- SceToSingleCellAssay(adata_, class = "SingleCellAssay")
    print("Dimensions before subsetting:")
    print(dim(sca))
    print("")
    # keep genes that are expressed in more than 5% of all cells
    sca <- sca[freq(sca)>0.05,]
    print("Dimensions after subsetting:")
    print(dim(sca))
    print("")
    # add a column to the data which contains scaled number of genes that are expressed in each cell
    cdr2 <- colSums(assay(sca)>0)
    colData(sca)$ngeneson <- scale(cdr2)
    # store the columns that we are interested in as factors
    label <- factor(colData(sca)$condition)
    # set the reference level
    label <- relevel(label,"control")
    print(levels(label))
    colData(sca)$label <- label
    # define and fit the model
    zlmCond <- zlm(~label + ngeneson, sca)
    #only test the condition coefficient.
    target_name <- paste0("label",levels(label)[2])
    summaryCond <- summary(zlmCond, doLRT=target_name)
    # summarize results
    summaryDt <- summaryCond$datatable
    fcHurdle <- merge(summaryDt[contrast==target_name & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
                      summaryDt[contrast==target_name & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients
    fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    result <- fcHurdle[fcHurdle$fdr<0.05,, drop=F]
    result <- stats::na.omit(as.data.frame(result))
    return(result)
}
#######

data_path = "data/SM2018_Tcells_processed.h5ad"
adata = sc.read_h5ad(data_path)
adata_unstim = adata[adata.obs.state == 'NoStim']
adata_stim = adata[adata.obs.state == 'Stim']
pert_list = list(adata.obs.condition.cat.categories)

start_time = time.time()
# stim
MAST_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_stim[adata_stim.obs.condition.isin([pert,'control'])]
    adata_subset = prep_anndata(adata_subset)
    # load in adata
    with localconverter(default_converter + pandas2ri.converter + anndata2ri.converter):
        %R -i adata_subset
    # run MAST
    %R res <- find_de_MAST_RE(adata_subset)
    # get res from R
    %R -o res
    MAST_degs_df_list.append(res)
    print('Finish {}'.format(pert))
with open("SM2018_Tcells/stim/MAST_degs_df_list.pkl", "wb") as f:
    pickle.dump(MAST_degs_df_list, f)

# unstim
MAST_degs_df_list = []
for pert in pert_list:
    adata_subset = adata_unstim[adata_unstim.obs.condition.isin([pert,'control'])]
    adata_subset = prep_anndata(adata_subset)
    # load in adata
    with localconverter(default_converter + pandas2ri.converter + anndata2ri.converter):
        %R -i adata_subset
    # run MAST
    %R res <- find_de_MAST_RE(adata_subset)
    # get res from R
    %R -o res
    MAST_degs_df_list.append(res)
    print('Finish {}'.format(pert))
with open("SM2018_Tcells/unstim/MAST_degs_df_list.pkl", "wb") as f:
    pickle.dump(MAST_degs_df_list, f)

end_time = time.time()
print("Running time: {:.2f} seconds".format(end_time - start_time))