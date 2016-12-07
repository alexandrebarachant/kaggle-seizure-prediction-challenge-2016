library(xgboost)
library(Matrix)
library("R.matlab")
library(caret)
set.seed(1234)
testid <- NULL
testpreds <- NULL
testpreds1 <- NULL
for (i in 1:3) {
  cat('\n',i)
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
  
  cat('model1...')
  set.seed(1234)
  model1 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1000)
  cat('model2...')
  set.seed(23)
  model2 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1000)
  cat('model3...')
  set.seed(13123)
  model3 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1000)
  set.seed(123131)
  cat('model4...')
  model4 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1000)
  cat('model5...')
  set.seed(1123123134)
  model5 <- xgb.train(   params              = param, 
                         data                = dtrain, 
                         nrounds             =  1000)
  save('model1','model2','model3','model4','model5', file = paste('./models/xgb_models_pat_',i,sep=""));  
  data$newF.T[is.na(data$newF.T)]   <- 0
  test <- xgb.DMatrix(data=data$newF.T)
  dec1 <- cbind(predict(model1, test),predict(model2, test),predict(model3, test),predict(model4, test),predict(model5, test))
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
write.csv(submission, "../submissions/Andriy_submissionXGB7_5mean.csv", row.names = F)