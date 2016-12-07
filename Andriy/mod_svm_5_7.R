library(xgboost)
library(randomForest)
library(Matrix)
library("R.matlab")
library(caret)
set.seed(1234)
testid <- NULL
testpreds1 <- NULL
testpreds2 <- NULL
testpreds3 <- NULL
library(glmnet)
library(LiblineaR)
library(MASS)

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
                         nrounds             =  1000
  )
  importance <- xgb.importance(model = model1)
  data$newF.T[is.na(data$newF.T)]   <- 0
  
  s=scale(train[,as.numeric(importance$Feature[1:300])],center=TRUE,scale=TRUE); 
  model=LiblineaR(data=s,target=as.factor(train.y),type=5,cost = 100, bias=TRUE,verbose=FALSE); 
  s2=scale(data$newF.T[,as.numeric(importance$Feature[1:300])],attr(s,"scaled:center"),attr(s,"scaled:scale"))
  dec1_2 = predict(model,s2,decisionValues=TRUE);
  dec1_2 =  1/(1 + exp(-dec1_2$decisionValues[,1]))
  save('model','importance', file = paste('./models/svm_model_pat_',i,sep=""));  
  
  myddd <- matrix(0, 1, nrow(data$newF.T))
  myddd [data$goodidx] = 1;
  myddd <- matrix(myddd, nrow = 19)
  myddd <- colMeans(myddd); myddd = which(myddd == 0);
  
  
  dec = matrix(0,1,length(dec1_2)); dec[data$goodidx] = dec1_2[data$goodidx];
  dec <- matrix(dec, nrow = 19)
  dectmp <- apply(dec,2,function(x) max(x))
  dectmp[myddd] <- 0 
  dectmp[which(is.na(dectmp))] <-0
  
  dec <- dectmp
  dec[myddd] <- 0;
  testpreds2 = c(testpreds2,dec)
  
}
submission <- data.frame(File=testid, Class=testpreds2)
cat("saving the submission file\n")
write.csv(submission, "../submissions/Andriy_submission5_7_SVM.csv", row.names = F)
