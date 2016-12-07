library(xgboost)
library(randomForest)
library(Matrix)
library("R.matlab")
library(caret)
set.seed(1234)
testid <- NULL
testpreds1 <- NULL
library(glmnet)
library(LiblineaR)


for (i in 1:3) {
  data = readMat(paste('train_ALL',i,'.mat', sep = ''))
  data$datalabels[which(data$datalabels==-1)]=0
  train.y <- data$datalabels
  train = rbind(data$newF.P,data$newF.P1,data$newF.I)
  dtrain <- xgb.DMatrix(data=train, label=train.y)
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = 0.3,
                  max_depth           = 3,
                  subsample           = 0.8,
                  colsample_bytree    = 1,
                  num_parallel_tree   = 2
  )
  
  model1 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1500
  )
  importance <- xgb.importance(model = model1)
  
  
  data$newF.T[is.na(data$newF.T)]   <- 0

  glmmod = glmnet(x=as.matrix(train[,as.numeric(importance$Feature[1:300])]), y=train.y, alpha=1,family = 'binomial')
  dec1_4 = predict(glmmod,as.matrix(data$newF.T[,as.numeric(importance$Feature[1:300])]),s=0.0001,type="response")
  save('glmmod', 'importance', file = paste('glm_models300_pat_',i,sep="")); 
  
  myddd <- matrix(0, 1, nrow(data$newF.T))
  myddd [data$goodidx] = 1;
  myddd <- matrix(myddd, nrow = 19)
  myddd <- colMeans(myddd); myddd = which(myddd == 0);
  
  dec = matrix(0,1,length(dec1_4)); dec[data$goodidx] = dec1_4[data$goodidx];
  dec <- matrix(dec, nrow = 19)
  dectmp <- apply(dec,2,function(x) max(x))
  dectmp[myddd] <- 0 
  dectmp[which(is.na(dectmp))] <-0
  
  testid = c(testid,paste('new_',i,'_',1:length(dectmp),'.mat',sep=''))
  testpreds1 = c(testpreds1,dectmp)
  
}
submission <- data.frame(File=testid, Class=testpreds1)
cat("saving the submission file\n")
write.csv(submission, "../submissions/submissionLR5_3_glmnet.csv", row.names = F)
