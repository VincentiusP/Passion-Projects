setwd("~/Documents/machine learning project")
install.packages('binaryLogic')
library('fastDummies')
library(devtools) 
library(xgboost) # Load XGBoost
library(caret) # Load Caret
library(OptimalCutpoints) # Load optimal cutpoints
library(ggplot2) # Load ggplot2
library(xgboostExplainer) # Load XGboost Explainer
library(pROC) # Load proc
library(SHAPforxgboost)


total_data <- read.csv('./Credit_Card_Approval_Data.csv', stringsAsFactors=T)

total_data <- dummy_cols(total_data,
           select_columns = c('Income_Type','Education_Type','Family_Status','Housing_Type','Job_Title'),
           remove_selected_columns = TRUE)
total_data$Applicant_Gender<-ifelse(total_data$Applicant_Gender=="M      ",1,0)
View(total_data)

total_obs <- dim(total_data)[1]
train_data_indices <- sample(1:total_obs, 0.8*total_obs)
train_data <- total_data[train_data_indices,]
test_data <- total_data[-train_data_indices,]

names(train_data)
sum(is.na(train_data))
sum(is.na(test_data))

summary(as.factor(train_data$Status))
summary(as.factor(test_data$Status))

dtrain <- xgb.DMatrix(data = as.matrix(train_data[, 1:54]), label = as.numeric(train_data$Status) -1)
# Create test matrix
dtest <- xgb.DMatrix(data = as.matrix(test_data[, 1:54]), label = as.numeric(test_data$Status) - 1)

set.seed(111111)
bst_1 <- xgboost(data = dtrain, # Set training data
                 
                 nrounds = 100, # Set number of rounds
                 
                 verbose = 1, # 1 - Prints out fit
                 print_every_n = 20, # Prints out result every 20th iteration
                 
                 objective = "binary:logistic", # Set objective
                 eval_metric = "auc",
                 eval_metric = "error") # Set evaluation metric to use

boost_preds <- predict(bst_1, dtrain) # Create predictions for xgboost model
# Join predictions and actual
pred_dat <- cbind.data.frame(boost_preds , train_data$class)
names(pred_dat) <- c("predictions", "response")
oc<- optimal.cutpoints(X = "predictions",
                       status = "response",
                       tag.healthy = 0,
                       data = pred_dat,
                       methods = "MaxEfficiency")

boost_preds_1 <- predict(bst_1, dtest) # Create predictions for xgboost model

pred_dat <- cbind.data.frame(boost_preds_1 , test_data$class)#
# Convert predictions to classes, using optimal cut-off
boost_pred_class <- rep(0, length(boost_preds_1))
boost_pred_class[boost_preds_1 >= oc$MaxEfficiency$Global$optimal.cutoff$cutoff[1]] <- 1


t <- table(boost_pred_class, test_data$class) # Create table
confusionMatrix(t, positive = "1") # Produce confusion matrix

set.seed(111111)
bst <- xgb.cv(data = dtrain, # Set training data
              
              nfold = 5, # Use 5 fold cross-validation
              
              eta = 0.1, # Set learning rate
              
              nrounds = 1000, # Set number of rounds
              early_stopping_rounds = 50, # Set number of rounds to stop at if there is no improvement
              
              verbose = 1, # 1 - Prints out fit
              nthread = 1, # Set number of parallel threads
              print_every_n = 20, # Prints out result every 20th iteration
              
              objective = "binary:logistic", # Set objective
              eval_metric = "auc",
              eval_metric = "error") # Set evaluation metric to use
