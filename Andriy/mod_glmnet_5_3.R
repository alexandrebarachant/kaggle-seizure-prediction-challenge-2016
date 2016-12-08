library(xgboost)
#library(randomForest)
library(Matrix)
library("R.matlab")
#library(caret)
set.seed(1234)
testid <- NULL
testpreds <- NULL
library(glmnet)
#library(LiblineaR)


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

  glmmod1 = glmnet(x=as.matrix(train[,as.numeric(importance$Feature[1:300])]), y=train.y, alpha=0,family = 'binomial')
  dec1_11 = predict(glmmod1,as.matrix(data$newF.T[,as.numeric(importance$Feature[1:300])]),s=0.0001,type="response")

  glmmod2 = glmnet(x=as.matrix(train[,as.numeric(importance$Feature[1:300])]), y=train.y, alpha=0,family = 'binomial')
  dec1_12 = predict(glmmod2,as.matrix(data$newF.T[,as.numeric(importance$Feature[1:300])]),s=0.0001,type="response")

  glmmod3 = glmnet(x=as.matrix(train[,as.numeric(importance$Feature[1:300])]), y=train.y, alpha=0,family = 'binomial')
  dec1_13 = predict(glmmod3,as.matrix(data$newF.T[,as.numeric(importance$Feature[1:300])]),s=0.0001,type="response")

  save('glmmod1','glmmod2','glmmod3', 'importance', file = paste('./models/glm_models300_pat_',i,sep="")); 
  
  dec1 <- cbind(dec1_11,dec1_12,dec1_13)
  dec2 <- apply(dec1, 2, function (x) apply(matrix(x,nrow = 19),2,function (x) max(x)))
  
  myddd <- matrix(0, 1, nrow(data$newF.T))
  myddd [data$goodidx] = 1;
  myddd <- matrix(myddd, nrow = 19)
  myddd <- colMeans(myddd); myddd = which(myddd == 0);

  dec2[myddd,] <- 0;
  testid = c(testid,paste('new_',i,'_',1:nrow(dec2),'.mat',sep=''))
  testpreds = rbind(testpreds,dec2)
 
}
dec3 <- apply(testpreds, 2, function(x)  rank(x,ties.method='average'))/(nrow(testpreds))
dec4 <- apply(dec3, 1, function(x)  mean(x))

submission <- data.frame(File=testid, Class=dec4)
cat("saving the submission file\n")
write.csv(submission, "../submissions/Andriy_submissionLR5_3_glmnet.csv", row.names = F)
