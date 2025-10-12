library(data.table)
library(tidyverse)
library(Matrix)
library(Seurat)
library(GSFA)
library(ggplot2)
start_time <- Sys.time()
data_dir <- "data/GSE119450_RAW/"
filename_tb <- 
  data.frame(experiment = c("D1S", "D2S", "D1N", "D2N"),
             prefix = c("GSM3375488_D1S", "GSM3375490_D2S", 
                        "GSM3375487_D1N", "GSM3375489_D2N"),
             stringsAsFactors = F)
seurat_lst <- list()
guide_lst <- list()

for (i in 1:4){
  experiment <- filename_tb$experiment[i]
  prefix <- filename_tb$prefix[i]
  cat(paste0("Loading data of ", experiment, " ..."))
  cat("\n\n")
  feature.names <- data.frame(fread(paste0(data_dir, experiment, "/genes.tsv"),
                                    header = FALSE), stringsAsFactors = FALSE)
  barcode.names <- data.frame(fread(paste0(data_dir, experiment, "/barcodes.tsv"),
                                    header = FALSE), stringsAsFactors = FALSE)
  barcode.names$V2 <- sapply(strsplit(barcode.names$V1, split = "-"),
                             function(x){x[1]})
  # Load the gene count matrix (gene x cell) and annotate the dimension names:
  dm <- readMM(file = paste0(data_dir, experiment, "/matrix.mtx"))
  rownames(dm) <- feature.names$V1
  colnames(dm) <- barcode.names$V2
  
  # Load the meta data of cells:
  metadata <- data.frame(fread(paste0(data_dir, experiment, "/",
                                      prefix, "_CellBC_sgRNA.csv.gz"),
                               header = T, sep = ','), check.names = F)
  metadata$gene_target <- sapply(strsplit(metadata$gRNA.ID, split = "[.]"),
                                 function(x){x[3]})
  metadata$guide <- sapply(strsplit(metadata$gRNA.ID, split = "[.]"),
                           function(x){paste0(x[2], ".", x[3])})
  
  metadata <- metadata %>% filter(Cell.BC %in% barcode.names$V2)
  targets <- unique(metadata$gene_target)
  targets <- targets[order(targets)]
  
  # Make a cell by perturbation matrix:
  guide_mat <- data.frame(matrix(nrow = nrow(metadata),
                                 ncol = length(targets)))
  rownames(guide_mat) <- metadata$Cell.BC
  colnames(guide_mat) <- targets
  for (m in targets){
    guide_mat[[m]] <- (metadata$gene_target == m) * 1
  }
  guide_lst[[experiment]] <- guide_mat
  
  # Only keep cells with gRNA info:
  dm.cells_w_gRNA <- dm[, metadata$Cell.BC]
  cat("Dimensions of final gene expression matrix: ")
  cat(dim(dm.cells_w_gRNA))
  cat("\n\n")
  
  dm.seurat <- CreateSeuratObject(dm.cells_w_gRNA, project = paste0("TCells_", experiment))
  dm.seurat <- AddMetaData(dm.seurat, metadata = guide_mat)
  seurat_lst[[experiment]] <- dm.seurat
}

combined_obj <- merge(seurat_lst[[1]], 
                      c(seurat_lst[[2]], seurat_lst[[3]], seurat_lst[[4]]),
                      add.cell.ids = filename_tb$experiment,
                      project = "T_cells_all_merged")

# QC
MT_genes <- feature.names %>% filter(startsWith(V2, "MT-")) %>% pull(V1)
combined_obj[['percent_mt']] <- PercentageFeatureSet(combined_obj, 
                                                     features = MT_genes)
combined_obj <- subset(combined_obj, 
                       subset = percent_mt < 10 & nFeature_RNA > 500)
VlnPlot(combined_obj, 
        features = c('nFeature_RNA', 'nCount_RNA', 'percent_mt'), 
        pt.size = 0)

# normalize
dev_res <- deviance_residual_transform(t(as.matrix(combined_obj@assays$RNA@counts)))

# HVGs & covariates
top_gene_index <- select_top_devres_genes(dev_res, num_top_genes = 6000)
dev_res_filtered <- dev_res[, top_gene_index]
covariate_df <- data.frame(lib_size = combined_obj$nCount_RNA,
                           umi_count = combined_obj$nFeature_RNA,
                           percent_mt = combined_obj$percent_mt)
dev_res_corrected <- covariate_removal(dev_res_filtered, covariate_df)
scaled.gene_exp <- scale(dev_res_corrected)

# input
sample_names <- colnames(combined_obj@assays$RNA@counts)
gene_names <- rownames(combined_obj@assays$RNA@counts)
rownames(scaled.gene_exp) <- sample_names
colnames(scaled.gene_exp) <- gene_names[top_gene_index]
G_mat <- combined_obj@meta.data[, 4:24]
G_mat <- as.matrix(G_mat)
num_cells <- colSums(G_mat)
num_cells_df <- data.frame(locus = names(num_cells),
                           count = num_cells)
sample_group <- combined_obj$orig.ident
sample_group <- (sample_group %in% c("TCells_D1S", "TCells_D2S")) * 1

# run
set.seed(92629)
fit0 <- fit_gsfa_multivar_2groups(Y = scaled.gene_exp, G = G_mat, 
                                  group = sample_group, K = 20,
                                  prior_type = "mixture_normal", 
                                  init.method = "svd",
                                  niter = 2000, used_niter = 1000,
                                  verbose = T, return_samples = T)
set.seed(92629)
fit <- fit_gsfa_multivar_2groups(Y = scaled.gene_exp, G = G_mat, 
                                 group = sample_group, fit0 = fit0,
                                 prior_type = "mixture_normal", 
                                 init.method = "svd",
                                 niter = 2000, used_niter = 1000,
                                 verbose = T, return_samples = T)
end_time <- Sys.time()
print(end_time - start_time)
