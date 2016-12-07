1. Run in Matlab the following 4 scripts (in any order or in parallel) to extract 4 features sets: 

FE_main_AR.m ---------- Autoregressive modelling errors (might take a while to extract)
FE_main_CONN.m -------- Connectivity features
FE_main_CSP_AR.m ------ Autoregressive modelling errors for 1 channels in CSP montage
FE_main_F.m ----------- Mainstream EEG feature from each channel 


2. Run in Matlab the following script to prepare the data for training: 

data_preprocess.m ----- Reads the data, imputes missing values, and saves the data for each patient

3. Run in R the following scripts to create models and submission files:

mod_glmnet_5_3.R ------ Creates GLM model and submission
mod_svm_5_7.R --------- Creates SVM model and submission
mod_xgb_7_5.R --------- Creates XGB model and submission
