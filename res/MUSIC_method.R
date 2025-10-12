library(Biostrings)
library(clusterProfiler)
library(devtools)
library(MUSIC)
library(Matrix)

counts <- readMM("data/SM2018_Tcells_counts_6000_stim.mtx")  
dim(counts)
gene_metadata <- read.csv("data/SM2018_Tcells_counts_6000_stim_gene_metadata.csv", row.names = 1)
perturb_information_df <- read.csv("data/SM2018_Tcells_counts_6000_stim_perturbation_info.csv", row.names =1)
head(gene_metadata)
expression_profile <- as.data.frame(as.matrix(t(counts)))
rm(counts)
gc()
rownames(expression_profile) <- rownames(gene_metadata)
colnames(expression_profile) <- rownames(perturb_information_df)
perturb_information<-as.character(perturb_information_df[,1])
names(perturb_information) <- rownames(perturb_information_df)
crop_seq_list<-Input_preprocess(expression_profile,perturb_information)
# We did not conduct data imputation, since it takes too long
# cell quality control
crop_seq_qc<-Cell_qc(crop_seq_list$expression_profile,
                     crop_seq_list$perturb_information,
                     gene_low=80,gene_high=10000,mito_high=0.3,
                     umi_low=100,umi_high=Inf,species="Hs",plot=F)
rm(crop_seq_list)
gc()
# obtain highly dispersion differentially expressed genes.
crop_seq_vargene<-Get_high_varGenes(crop_seq_qc$expression_profile,crop_seq_qc$perturb_information,plot=T)
rm(crop_seq_qc)
gc()

# get topics
cat("Fitting model with 5 topics ... \n")
system.time(
  topic_1 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 5))
saveRDS(topic_1, "SM2018_Tcells/stim/music_output/music_merged_5_topics.rds")
cat("Fitting model with 6 topics ... \n")
system.time(
  topic_2 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 6))
saveRDS(topic_2, "SM2018_Tcells/stim/music_output/music_merged_6_topics.rds")
cat("Fitting model with 7 topics ... \n")
system.time(
  topic_3 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 7))
saveRDS(topic_3, "SM2018_Tcells/stim/music_output/music_merged_7_topics.rds")
cat("Fitting model with 8 topics ... \n")
system.time(
  topic_4 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 8))
saveRDS(topic_4, "SM2018_Tcells/stim/music_output/music_merged_8_topics.rds")
## try larger numbers of topics
cat("Fitting model with 10 topics ... \n")
system.time(
  topic_5 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 10))
saveRDS(topic_5, "SM2018_Tcells/stim/music_output/music_merged_10_topics.rds")
cat("Fitting model with 15 topics ... \n")
system.time(
  topic_6 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 15))
saveRDS(topic_6, "SM2018_Tcells/stim/music_output/music_merged_15_topics.rds")
cat("Fitting model with 20 topics ... \n")
system.time(
  topic_7 <- Get_topics(crop_seq_vargene$expression_profile,
                        crop_seq_vargene$perturb_information,
                        topic_number = 20))
saveRDS(topic_7, "SM2018_Tcells/stim/music_output/music_merged_20_topics.rds")

## Pick optimal number of topics ####
topic_model_list <- list()
topic_model_list$models <- list()
topic_model_list$perturb_information <- topic_1$perturb_information
topic_model_list$models[[1]] <- topic_1$models[[1]]
topic_model_list$models[[2]] <- topic_2$models[[1]]
topic_model_list$models[[3]] <- topic_3$models[[1]]
topic_model_list$models[[4]] <- topic_4$models[[1]]
topic_model_list$models[[5]] <- topic_5$models[[1]]
topic_model_list$models[[6]] <- topic_6$models[[1]]
topic_model_list$models[[7]] <- topic_7$models[[1]]


# select the optimal topic number.  
optimalModel<-Select_topic_number(topic_model_list$models,plot=T,
                                  plot_path = "SM2018_Tcells/stim/music_output/select_topic_number_5to8to20.pdf")

topic_optimal_model<- readRDS("SM2018_Tcells/stim/music_output/music_merged_6_topics.rds")
topic_optimal_model$models[[1]]@terms

loading <- topic_optimal_model$models[[1]]@beta
colnames(loading) <- topic_optimal_model$models[[1]]@terms
factor_vals <- topic_optimal_model$models[[1]]@gamma
rownames(factor_vals) <- topic_optimal_model$models[[1]]@documents
write.csv(loading, file = "SM2018_Tcells/stim/music_output/loading.csv", row.names = TRUE)
write.csv(factor_vals, file = "SM2018_Tcells/stim/music_output/factor_vals.csv", row.names = TRUE)

# annotate each topic's functions. For parameter "species", Hs(homo sapiens) or Mm(mus musculus) are available.
topic_func<-Topic_func_anno(optimalModel,species="Hs",plot=T,
                            plot_path = "SM2018_Tcells/stim/music_output/GO.pdf")
write.csv(topic_func$topic_annotation_result, file = "SM2018_Tcells/stim/music_output/go_res.csv", row.names = TRUE)
# calculate topic distribution for each cell.
distri_diff<-Diff_topic_distri(optimalModel,topic_model_list$perturb_information,plot=T,
                               plot_path = "SM2018_Tcells/stim/music_output/distri_diff.pdf")
write.csv(distri_diff, file = "SM2018_Tcells/stim/music_output/distri_diff.csv", row.names = TRUE)
# calculate the overall perturbation effect ranking list.
rank_overall_result<-Rank_overall(distri_diff)
saveRDS(rank_overall_result, "SM2018_Tcells/stim/music_output/rank_overall_result.rds")
# calculate the topic-specific ranking list.
rank_topic_specific_result<-Rank_specific(distri_diff)
saveRDS(rank_topic_specific_result, "SM2018_Tcells/stim/music_output/rank_topic_specific_result.rds")
# calculate the perturbation correlation.
perturb_cor<-Correlation_perturbation(distri_diff,plot=F)
saveRDS(perturb_cor, "SM2018_Tcells/stim/music_output/perturb_cor.rds")


