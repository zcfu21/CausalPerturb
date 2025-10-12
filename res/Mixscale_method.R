library(Seurat)
library(ggridges)
library(ggplot2)
library(Mixscale)
library(Matrix)

start_time <- Sys.time()
############ stim
counts <- readMM("data/SM2018_Tcells_counts_6000_stim.mtx") 
counts <- t(counts)
cell_metadata <- read.csv("data/SM2018_Tcells_counts_6000_stim_cell_metadata.csv", row.names = 1)
gene_metadata <- read.csv("data/SM2018_Tcells_counts_6000_stim_gene_metadata.csv", row.names = 1)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)

# create Seurat object
seurat_obj <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)

# standard pre-processing
seurat_obj = NormalizeData(seurat_obj)
seurat_obj = FindVariableFeatures(seurat_obj)
seurat_obj = ScaleData(seurat_obj)
seurat_obj = RunPCA(seurat_obj)

# calculate Perturbation signatures 
seurat_obj <- CalcPerturbSig(
  object = seurat_obj, 
  assay = "RNA", 
  slot = "data", 
  gd.class ="target", 
  nt.cell.class = "NonTarget", 
  reduction = "pca", 
  ndims = 40, 
  num.neighbors = 20, 
  new.assay.name = "PRTB", 
  split.by = NULL)

# Mixscale
seurat_obj = RunMixscale(
  object = seurat_obj, 
  assay = "PRTB", 
  slot = "scale.data", 
  labels = "target", 
  nt.class.name = "NonTarget", 
  min.de.genes = 5, 
  logfc.threshold = 0.1,
  de.assay = "RNA",
  max.de.genes = 100, 
  new.class.name = "mixscale_score", 
  fine.mode = F, 
  verbose = F, 
  split.by = NULL)
RidgePlot(
  seurat_obj,
  features = "mixscale_score",
  group.by = "target") + NoLegend()
prtb_score <- Tool(object = seurat_obj, slot = "RunMixscale")

write.csv(seurat_obj@meta.data, paste0("SM2018_Tcells/stim/Mixscale_score", ".csv"))

# DEG
de_res = Run_wmvRegDE(object = seurat_obj, assay = "RNA", slot = "counts",
                      labels = "target", nt.class.name = "NonTarget",
                      PRTB_list = pert_list,
                      logfc.threshold = 0.2,
                      split.by = NULL)

for (i in seq_along(de_res)) {
  write.csv(de_res[[i]], paste0("SM2018_Tcells/stim/Mixscale_res/df_", i, ".csv"), row.names = FALSE)
}

############ unstim

counts <- readMM("data/SM2018_Tcells_counts_6000_unstim.mtx")  
counts <- t(counts)
cell_metadata <- read.csv("data/SM2018_Tcells_counts_6000_unstim_cell_metadata.csv", row.names = 1)
gene_metadata <- read.csv("data/SM2018_Tcells_counts_6000_unstim_gene_metadata.csv", row.names = 1)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)

# create Seurat object
seurat_obj <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)

# standard pre-processing
seurat_obj = NormalizeData(seurat_obj)
seurat_obj = FindVariableFeatures(seurat_obj)
seurat_obj = ScaleData(seurat_obj)
seurat_obj = RunPCA(seurat_obj)

# calculate Perturbation signatures 
seurat_obj <- CalcPerturbSig(
  object = seurat_obj, 
  assay = "RNA", 
  slot = "data", 
  gd.class ="target", 
  nt.cell.class = "NonTarget", 
  reduction = "pca", 
  ndims = 40, 
  num.neighbors = 20, 
  new.assay.name = "PRTB", 
  split.by = NULL)

# Mixscale
seurat_obj = RunMixscale(
  object = seurat_obj, 
  assay = "PRTB", 
  slot = "scale.data", 
  labels = "target", 
  nt.class.name = "NonTarget", 
  min.de.genes = 5, 
  logfc.threshold = 0.1,
  de.assay = "RNA",
  max.de.genes = 100, 
  new.class.name = "mixscale_score", 
  fine.mode = F, 
  verbose = F, 
  split.by = NULL)
RidgePlot(
  seurat_obj,
  features = "mixscale_score",
  group.by = "target") + NoLegend()
prtb_score <- Tool(object = seurat_obj, slot = "RunMixscale")

write.csv(seurat_obj@meta.data, paste0("SM2018_Tcells/unstim/Mixscale_score", ".csv"))

# DEG
de_res = Run_wmvRegDE(object = seurat_obj, assay = "RNA", slot = "counts",
                      labels = "target", nt.class.name = "NonTarget",
                      PRTB_list = pert_list,
                      logfc.threshold = 0.2,
                      split.by = NULL)

for (i in seq_along(de_res)) {
  write.csv(de_res[[i]], paste0("SM2018_Tcells/unstim/Mixscale_res/df_", i, ".csv"), row.names = FALSE)
}

end_time <- Sys.time()
print(end_time - start_time)
