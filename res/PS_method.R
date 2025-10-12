library(Seurat)
library(ggplot2)
library(patchwork)
library(scales)
library(dplyr)
library(scMAGeCK)
library(Matrix)
packageVersion("scMAGeCK")

start_time <- Sys.time()
counts <- readMM("data/SM2018_Tcells_counts_6000_all.mtx")  
counts <- t(counts)
cell_metadata <- read.csv("data/SM2018_Tcells_counts_6000_all_cell_metadata.csv", row.names = 1)
gene_metadata <- read.csv("data/SM2018_Tcells_counts_6000_all_gene_metadata.csv", row.names = 1)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)

# Seurat object
seurat_obj <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
seurat_obj <- NormalizeData(object = seurat_obj) %>% FindVariableFeatures() %>% ScaleData()
bc_frame <- read.csv('data/PS_bc_frame.csv',
                     row.names = 1)
seurat_obj<-assign_cell_identity(bc_frame, seurat_obj)
pert_list <- c('ARID1A','BTLA','C10orf54','CBLB','CD3D','CD5','CDKN1B','DGKA',
               'DGKZ','HAVCR2','LAG3','LCP2','MEF2D','PDCD1','RASA2','SOCS1',
               'STAT6','TCEB2','TMEM222','TNFRSF9')

eff_object <- scmageck_eff_estimate(seurat_obj, bc_frame, perturb_gene=pert_list, 
                                    non_target_ctrl = 'NonTarget',subset_rds = T, 
                                    lambda = 0, target_gene_max = 100)
eff_estimat=eff_object$eff_matrix
write.csv(eff_estimat, paste0("SM2018_Tcells/stim/PS_score_all", ".csv"))
end_time <- Sys.time()
print(end_time - start_time)









