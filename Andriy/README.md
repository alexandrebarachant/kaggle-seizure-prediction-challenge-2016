* Run in Matlab the following 4 scripts (in any order or in parallel) to extract 4 feature sets:

Autoregressive modelling errors (might take a while to extract):

FE_main_AR.m 

Connectivity features:

FE_main_CONN.m

Autoregressive modelling errors for 1 channels in CSP montage:

FE_main_CSP_AR.m 

Mainstream EEG feature from each channel:

FE_main_F.m


* Run in Matlab the following script to prepare the data for training (read the data, impute missing values, and save the data for each patient): 

data_preprocess.m 

* Run in R the following scripts to create models and submission files:

Creates GLM model and submission

mod_glmnet_5_3.R

Creates SVM model and submission
 
mod_svm_5_7.R

Creates XGB model and submission

mod_xgb_7_5.R 
