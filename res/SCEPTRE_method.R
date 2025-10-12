library(sceptre)
library(Matrix)
start_time <- Sys.time()

############### stim
# 0 prepare data
A<-readMM("data/SM2018_Tcells_counts_6000_stim.mtx")
A[1:10,1:10]
grna_matrix <- read.csv('data/SM2018_Tcells_counts_6000_stim_grna_matrix.csv',
                        row.names = 1)
grna_matrix[1:10,1:10]
response_names <- read.csv('data/SM2018_Tcells_counts_6000_stim_response_names.csv',
                           row.names = 1)
response_names <- response_names$gene_name
A <- t(A)
rownames(A) <- response_names
colnames(A) <- colnames(grna_matrix)
A[1:10,1:10]
grna_matrix <- as.matrix(grna_matrix)*50
grna_matrix <- as(grna_matrix, "TsparseMatrix")
grna_matrix[1:10,1:10]

grna_target_data_frame <- read.csv('data/SM2018_Tcells_counts_6000_stim_grna_target_data_frame.csv',
                                   row.names = 1)
grna_target_data_frame

# 1 import data
sceptre_object_lowmoi <- import_data(
  response_matrix = A,
  grna_matrix = grna_matrix,
  grna_target_data_frame = grna_target_data_frame,
  response_names = response_names,
  moi = "low"
)
sceptre_object_lowmoi

# 2 set analysis params
positive_control_pairs_lowmoi <- construct_positive_control_pairs(
  sceptre_object = sceptre_object_lowmoi
)
head(positive_control_pairs_lowmoi) # low MOI CRISPRko dataset

discovery_pairs_lowmoi <- construct_trans_pairs(
  sceptre_object = sceptre_object_lowmoi,
  positive_control_pairs = positive_control_pairs_lowmoi,
  pairs_to_exclude = "pc_pairs"
)
head(discovery_pairs_lowmoi)

# low-MOI CRISPRko data
sceptre_object_lowmoi <- set_analysis_parameters(
  sceptre_object = sceptre_object_lowmoi,
  discovery_pairs = discovery_pairs_lowmoi,
  positive_control_pairs = positive_control_pairs_lowmoi)
print(sceptre_object_lowmoi)

# 3 assign gRNAs
sceptre_object_lowmoi_thresholding <- assign_grnas(
  sceptre_object = sceptre_object_lowmoi,
  method = "thresholding"
)

sceptre_object_lowmoi <- sceptre_object_lowmoi_thresholding

print(sceptre_object_lowmoi)

# 4 Run QC
sceptre_object_lowmoi <- run_qc(
  sceptre_object = sceptre_object_lowmoi)
print(sceptre_object_lowmoi)

# 5 Run Calibration check
sceptre_object_lowmoi <- run_calibration_check(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)
# plot(sceptre_object_lowmoi)

# 6 Run power check
sceptre_object_lowmoi <- run_power_check(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)

# plot(sceptre_object_lowmoi)
# 7 Run DEG
sceptre_object_lowmoi <- run_discovery_analysis(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)

# 8 Get result
discovery_result <- get_result(
  sceptre_object = sceptre_object_lowmoi,
  analysis = "run_discovery_analysis"
)
head(discovery_result)

write.csv(discovery_result, "SM2018_Tcells/stim/sceptre_res.csv")   
# summary
print(sceptre_object_lowmoi)

############### unstim
# 0 prepare data
A<-readMM("data/SM2018_Tcells_counts_6000_unstim.mtx")
A[1:10,1:10]
grna_matrix <- read.csv('data/SM2018_Tcells_counts_6000_unstim_grna_matrix.csv',
                        row.names = 1)
grna_matrix[1:10,1:10]
response_names <- read.csv('data/SM2018_Tcells_counts_6000_unstim_response_names.csv',
                           row.names = 1)
response_names <- response_names$gene_name
A <- t(A)
rownames(A) <- response_names
colnames(A) <- colnames(grna_matrix)
A[1:10,1:10]
grna_matrix <- as.matrix(grna_matrix)*50
grna_matrix <- as(grna_matrix, "TsparseMatrix")
grna_matrix[1:10,1:10]

grna_target_data_frame <- read.csv('data/SM2018_Tcells_counts_6000_unstim_grna_target_data_frame.csv',
                                   row.names = 1)
grna_target_data_frame

# 1 import data
sceptre_object_lowmoi <- import_data(
  response_matrix = A,
  grna_matrix = grna_matrix,
  grna_target_data_frame = grna_target_data_frame,
  response_names = response_names,
  moi = "low"
)
sceptre_object_lowmoi

# 2 set analysis params
positive_control_pairs_lowmoi <- construct_positive_control_pairs(
  sceptre_object = sceptre_object_lowmoi
)
head(positive_control_pairs_lowmoi) # low MOI CRISPRko dataset

discovery_pairs_lowmoi <- construct_trans_pairs(
  sceptre_object = sceptre_object_lowmoi,
  positive_control_pairs = positive_control_pairs_lowmoi,
  pairs_to_exclude = "pc_pairs"
)
head(discovery_pairs_lowmoi)

# low-MOI CRISPRko data
sceptre_object_lowmoi <- set_analysis_parameters(
  sceptre_object = sceptre_object_lowmoi,
  discovery_pairs = discovery_pairs_lowmoi,
  positive_control_pairs = positive_control_pairs_lowmoi)
print(sceptre_object_lowmoi)

# 3 assign gRNAs
sceptre_object_lowmoi_thresholding <- assign_grnas(
  sceptre_object = sceptre_object_lowmoi,
  method = "thresholding"
)

sceptre_object_lowmoi <- sceptre_object_lowmoi_thresholding

print(sceptre_object_lowmoi)

# 4 Run QC
sceptre_object_lowmoi <- run_qc(
  sceptre_object = sceptre_object_lowmoi)
print(sceptre_object_lowmoi)

# 5 Run Calibration check
sceptre_object_lowmoi <- run_calibration_check(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)
# plot(sceptre_object_lowmoi)

# 6 Run power check
sceptre_object_lowmoi <- run_power_check(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)

# plot(sceptre_object_lowmoi)
# 7 Run DEG
sceptre_object_lowmoi <- run_discovery_analysis(
  sceptre_object = sceptre_object_lowmoi,
  parallel = FALSE
)

# 8 Get result
discovery_result <- get_result(
  sceptre_object = sceptre_object_lowmoi,
  analysis = "run_discovery_analysis"
)
head(discovery_result)

write.csv(discovery_result, "SM2018_Tcells/unstim/sceptre_res.csv")   

# summary
print(sceptre_object_lowmoi)

end_time <- Sys.time()
print(end_time - start_time)



