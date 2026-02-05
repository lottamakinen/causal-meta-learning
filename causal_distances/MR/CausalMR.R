library(data.table)
library(tidyverse)
library(lubridate)
library(dplyr)
library(readxl)
library(TwoSampleMR)


################################################################################################################################
#################################################### ENDPOINT OF INTEREST ######################################################
################################################################################################################################

# Target endpoint for MR analysis
endpoint <- "C_STROKE"

# Path configurations (update these for your environment)
endInfoDir <- "path/to/endpoint_info.xlsx"
endpointPathlong <- "path/to/endpoint_longitudinal.txt.gz"

######## Mendelian Randomization with fine-mapping results ###############

# Get all summary statistics files
files <- list.files(path="path/to/summary_stats/release", pattern="*.gz", full.names=TRUE, recursive=TRUE)
myfilesfinished <- files[!grepl(".tbi", files)]

# Get fine-mapping result files
condFiles <- list.files(path="path/to/finemap/summary", pattern="*.SUSIE_99.cred.summary.tsv", full.names=TRUE, recursive=TRUE)

endpointlist <- c(endpoint)

# Process summary statistics for the outcome phenotype
for (endpoint in endpointlist){
  endpointsumstatspath <- myfilesfinished[grepl(endpoint, myfilesfinished)]
  endpoiSumStat <- fread(cmd=paste("zcat", endpointsumstatspath))
  
  # Add columns needed for MR
  endpoiSumStat$pheno <- endpoint
  names(endpoiSumStat)[1] <- "chr"
  endpoiSumStat$samplesize <- 453733  # Update with your sample size
  endpoiSumStat$SNPID <- paste0("chr", endpoiSumStat$chr, "_", endpoiSumStat$pos, "_", endpoiSumStat$ref, "_", endpoiSumStat$alt)
  endpoiSumStat <- as.data.frame(endpoiSumStat)
  
  # Format as outcome data for TwoSampleMR
  endpoiSumStat1 <- format_data(
    endpoiSumStat,
    type = "outcome",
    phenotype_col = "pheno",
    snps = NULL,
    header = TRUE,
    snp_col = "SNPID",
    beta_col = "beta",
    se_col = "sebeta",
    samplesize_col = "samplesize",
    eaf_col = "af_alt",
    effect_allele_col = "alt",
    other_allele_col = "ref",
    pval_col = "pval",
    gene_col = "nearest_genes",
    chr_col = "chr",
    pos_col = "pos",
    log_pval = FALSE
  )

  # Create results table
  endTable <- data.frame(matrix(ncol=7))
  names(endTable) <- c("Endpoint", "Exposure", "MR_Egger", "IVW", "WeightedMode", "correct_causal_direction", "steiger_pval")
  
  # Loop through each fine-mapping result as exposure
  for (condpath in condFiles){
    # Get exposure endpoint name
    endpoinname <- tail(strsplit(condpath, split="/|.SUSIE_99.cred.summary.tsv")[[1]], 1)
    
    # Read exposure file
    expoSumStat <- fread(condpath)
    
    # Parse variant information
    expoSumStat$CHR <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,1]
    expoSumStat$BP <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,2]
    expoSumStat$A1 <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,3]
    expoSumStat$A2 <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,4]
    
    expoSumStat$samplesize <- 453733  # Update with your sample size
    expoSumStat <- as.data.frame(expoSumStat)
    
    # Format as exposure data for TwoSampleMR
    expoSumStat1 <- format_data(
      expoSumStat,
      phenotype_col = "trait",
      type = "exposure",
      snps = NULL,
      header = TRUE,
      snp_col = "rsid",
      beta_col = "beta",
      se_col = "sd",
      eaf_col = "prob",
      effect_allele_col = "A2",
      other_allele_col = "A1",
      pval_col = "p",
      chr_col = "CHR",
      pos_col = "BP",
      log_pval = FALSE
    )
    
    # Harmonise and run MR
    dat <- harmonise_data(expoSumStat1, endpoiSumStat1)
    res <- mr(dat)
    out <- directionality_test(dat)
    
    # Store results
    if(nrow(res) == 0){
      endTable <- rbind(endTable, c(endpoint, endpoinname, NaN, NaN, NaN, NaN, NaN))
    } else {
      if (is.null(out)){
        endTable <- rbind(endTable, c(endpoint, endpoinname, res$pval[1], res$pval[2], res$pval[3], NaN, NaN))
      } else {
        endTable <- rbind(endTable, c(endpoint, endpoinname, res$pval[1], res$pval[2], res$pval[3], out$correct_causal_direction[1], out$steiger_pval[1]))
      }
    }
  }
  fwrite(endTable, file = paste0("output/MR_table_", endpoint, "_finemap.txt"))
}


######## Pairwise MR for top hits ###############

# Load previous results and filter significant associations
endpoiSumStat <- fread("output/MR_table_C_STROKE_finemap.txt")
endpoiSumStat <- endpoiSumStat[order(IVW)]
endpoiSumStat <- endpoiSumStat[IVW < 1e-10]

# Get file lists
files <- list.files(path="path/to/summary_stats/release", pattern="*.gz", full.names=TRUE, recursive=TRUE)
myfilesfinished <- files[!grepl(".tbi", files)]

condFiles <- list.files(path="path/to/finemap/summary", pattern="*.*_99.cred.summary.tsv", full.names=TRUE, recursive=TRUE)

endpointlist <- endpoiSumStat$Exposure

# Pairwise MR between all significant exposures
for (endpoint in endpointlist){
  endpointsumstatspath <- myfilesfinished[grepl(endpoint, myfilesfinished)]
  endpoiSumStat <- fread(cmd=paste("zcat", endpointsumstatspath))
  
  endpoiSumStat$pheno <- endpoint
  names(endpoiSumStat)[1] <- "chr"
  endpoiSumStat$samplesize <- 453733
  endpoiSumStat$SNPID <- paste0("chr", endpoiSumStat$chr, "_", endpoiSumStat$pos, "_", endpoiSumStat$ref, "_", endpoiSumStat$alt)
  endpoiSumStat <- as.data.frame(endpoiSumStat)
  
  endpoiSumStat1 <- format_data(
    endpoiSumStat,
    type = "outcome",
    phenotype_col = "pheno",
    snps = NULL,
    header = TRUE,
    snp_col = "SNPID",
    beta_col = "beta",
    se_col = "sebeta",
    samplesize_col = "samplesize",
    eaf_col = "af_alt",
    effect_allele_col = "alt",
    other_allele_col = "ref",
    pval_col = "pval",
    gene_col = "nearest_genes",
    chr_col = "chr",
    pos_col = "pos",
    log_pval = FALSE
  )
  
  # Create results table
  endTable <- data.frame(matrix(ncol=7))
  names(endTable) <- c("Endpoint", "Exposure", "MR_Egger", "IVW", "WeightedMode", "correct_causal_direction", "steiger_pval")
  
  for (subendpoint in endpointlist){
    condpath <- condFiles[grepl(paste0(subendpoint, ".SUSIE"), condFiles)]
    
    expoSumStat <- fread(condpath)
    
    expoSumStat$CHR <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,1]
    expoSumStat$BP <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,2]
    expoSumStat$A1 <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,3]
    expoSumStat$A2 <- matrix(unlist(strsplit(expoSumStat$v, split=":")), ncol=4, byrow=T)[,4]
    
    expoSumStat$samplesize <- 453733
    expoSumStat <- as.data.frame(expoSumStat)
    
    expoSumStat1 <- format_data(
      expoSumStat,
      phenotype_col = "trait",
      type = "exposure",
      snps = NULL,
      header = TRUE,
      snp_col = "rsid",
      beta_col = "beta",
      se_col = "sd",
      eaf_col = "prob",
      effect_allele_col = "A2",
      other_allele_col = "A1",
      pval_col = "p",
      chr_col = "CHR",
      pos_col = "BP",
      log_pval = FALSE
    )
    
    dat <- harmonise_data(expoSumStat1, endpoiSumStat1)
    res <- mr(dat)
    out <- directionality_test(dat)
    
    # Store results
    if(nrow(res) == 0){
      endTable <- rbind(endTable, c(endpoint, subendpoint, NaN, NaN, NaN, NaN, NaN))
    } else {
      if (is.null(out)){
        endTable <- rbind(endTable, c(endpoint, subendpoint, res$pval[1], res$pval[2], res$pval[3], NaN, NaN))
      } else {
        endTable <- rbind(endTable, c(endpoint, subendpoint, res$pval[1], res$pval[2], res$pval[3], out$correct_causal_direction[1], out$steiger_pval[1]))
      }
    }
  }
  fwrite(endTable, file = paste0("output/MR_table_", endpoint, "_pairwise_finemap.txt"))
}