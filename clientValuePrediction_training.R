## Client value prediction 
## 

# Preample
#close the ODBC connection
odbcClose(con)
rm(list = ls(all.names = TRUE))

#Install commands

#install.packages("devtools", dependencies = TRUE)
#install.packages("RODBC", dependencies = TRUE)
#install.packages("data.table", dependencies = TRUE)
#install.packages("caTools", dependencies = TRUE)
#install.packages("xtable", dependencies = TRUE)
#install.packages("dplyr", dependencies = TRUE)
#install.packages("formattable", dependencies = TRUE)
#install.packages("RcppRoll", dependencies = TRUE)
#install.packages("caret", dependencies = TRUE)
#install.packages("sqldf", dependencies = TRUE)
#install.packages("xgboost", dependencies = TRUE)
#install.packages("bit64", dependencies = TRUE)


#Libraries
library(devtools)
library(odbc)
library(DBI)
library(data.table)
library(caTools)
library(xtable)
library(dplyr)
library(formattable)
library(RcppRoll)
library(caret)
library(xgboost)
library(bit64)

options(scipen = 999)


setwd("C:/Users/carl_/Dropbox/Consulting/Machine Learning projekter/Dacapo")
getwd()

# Build connection and import dataset:

DBI_Connection <- dbConnect(odbc(), .connection_string = "Driver={SQL Server};server=;UID=;PWD=;database=", timeout = 30)
DBI_Connection
# Querying full dataset
#--------------------------------------------------------------------------------------------------------
dt_fullData <- dbFetch(dbSendQuery(DBI_Connection, 
"DATA QUERY GOES HERE"))

dt_Dates <- dbFetch(dbSendQuery(DBI_Connection, 
"DATE QUERY GOES HERE"))
dt_fullData$StartWeekDate <- as.character(dt_fullData$StartWeekDate)
dt_Dates$Date <- as.character(dt_Dates$Date)
dt_fullData_ <- merge(dt_fullData, dt_Dates, by.x = "StartWeekDate", by.y = "Date", all.x = TRUE)

dt_fullData <- subset(dt_fullData_, select = -c(StartWeekDate, Month.x))
# Remove unmerged data set
rm("dt_fullData_")
#--------------------------------------------------------------------------------------------------------
# Data formats and wrangling
#--------------------------------------------------------------------------------------------------------

<Insert required data wrangling>

#---------------------------------------------------------------------------------------------------------
# Teaching the algorithm temporal awareness
#---------------------------------------------------------------------------------------------------------

<Insert calculated time-related metrics like moving averages, gradients, etc> 

#---------------------------------------------------------------------------------------------------------
# Splitting the dataset into the training and the test set - split på Counterparts
#---------------------------------------------------------------------------------------------------------
dt_distinctCounterparts <- unique(dt_fullData[c("CustomerId")])
split = sample.split(dt_distinctCounterparts$CustomerId, SplitRatio = 0.85)

training_set_ = data.table(na.omit(subset(dt_distinctCounterparts, split == TRUE)))
training_set <- data.table(inner_join(dt_fullData, training_set_ , by = ("CustomerId")))

test_set_ =     data.table(na.omit(subset(dt_distinctCounterparts, split == FALSE)))
test_set <-     data.table(inner_join(dt_fullData, test_set_ , by = ("CustomerId")))
rm("dt_fullData")

# removing single value factor columns

training_set4 <- select(training_set, -c('TREfuture12')) 
training_set12 <- select(training_set, -c('TREfuture4'))

test_set4 <- select(test_set, -c('TREfuture12'))
test_set12 <- select(test_set, -c('TREfuture4'))


# Building the two boosting matrix (4 week and 12 week)
#First the 4 week forecast ...
training_setY4w = training_set4$TREfuture4
training_setX4w <- model.matrix(TREfuture4~ . -1, data = training_set4)
Xmatrix4w = xgb.DMatrix(data = training_setX4w, label = training_setY4w)

test_set4Yw = test_set4$TREfuture4
test_set4Xw <- model.matrix(TREfuture4~ . -1, data = test_set4)
Xmatrix4w_t = xgb.DMatrix(data = test_set4Xw, label = test_set4Yw)

# ... and then the 12 week forecast
training_setY12w = training_set12$TREfuture12
training_setX12w <- model.matrix(TREfuture12~ . -1, data = training_set12)
Xmatrix12w = xgb.DMatrix(data = training_setX12w, label = training_setY12w)

test_set12Yw = test_set12$TREfuture12
test_set12Xw <- model.matrix(TREfuture12~ . -1, data = test_set12)
Xmatrix12w_t = xgb.DMatrix(data = test_set12Xw, label = test_set12Yw)

rm("training_set")
rm("test_set")
rm("training_set_")
rm("test_set_")

# ------------------------------------------------------------------------------------------------------
# Gridsearch hyperpameters
# ------------------------------------------------------------------------------------------------------

#tuneGrid <- expand.grid(max_depth = c(11, 12, 13),
#                        eta = c(0.1)
#                        )

#ntrees <- 100 

#system.time(rmseErrorsHyperparameters <- apply(tuneGrid, 1, function(parameterList){

#Extract Parameters to test
#    currentDepth <- parameterList[["max_depth"]]
#    currentEta <- parameterList[["eta"]]
    
#    xgboostModelCV <- xgb.cv(data =  Xmatrix12w, nrounds = ntrees, nfold = 5, showsd = TRUE, 
#                             metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
#                             "objective" = "reg:squarederror", "max.depth" = currentDepth, "eta" = currentEta,                               
#                             print_every_n = 10, booster = "gbtree",
#                             early_stopping_rounds = 10)
    
#    xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
#    rmse <- tail(xvalidationScores$test_rmse_mean, 1)
#    trmse <- tail(xvalidationScores$train_rmse_mean,1)
#    output <- return(c(rmse, trmse, currentDepth, currentEta))}))

#output <- as.data.frame(t(rmseErrorsHyperparameters))
#varnames <- c("TestRMSE", "TrainRMSE", "Depth", "eta")
#names(output) <- varnames
#head(output)
#--------------------------------------------------------------------------------------------------------------
params = list(
  booster="gbtree",
  eta=0.1,
  max_depth=13,
  gamma=2,
  subsample=0.8,
  colsample_bytree=0.7,
  objective = "reg:squarederror",
  eval_metric="rmse"
)

#---------------------------------------------------------------------------------------------------------
# Training the algorithm for variable 4 weeks 
#---------------------------------------------------------------------------------------------------------
nround4w = 851  #number rounds of ensemble
version4w = 2   #version number of model used
fileExtension4w = ".dat"
modelName4w = sprintf("xgboost4w_%s_v%s%s", nround4w, version4w, fileExtension4w) 

regressor4w = xgb.train(params = params,
                      data = Xmatrix4w, 
                      nround = nround4w,
                      early_stopping_rounds=25,
                      watchlist=list(val1=Xmatrix4w,val2=Xmatrix4w_t),
                      print_every_n = 25,
                      verbose = 1)

#Saves the regressor intelligence. Model can then be used without retraining algorithm.
# save the model to disk
saveRDS(regressor4w, "modelName4w")

turnover_pred4w = predict(regressor4w, Xmatrix4w_t)
test_result4 <- data.table(test_set4, turnover_pred4w)

#---------------------------------------------------------------------------------------------------------
# Training the algorithm for variable 12 weeks
#---------------------------------------------------------------------------------------------------------
nround12w = 851 #number rounds of ensemble
version12w = 2   #version number of model used
fileExtension12w = ".dat"
modelName12w = sprintf("xgboost12w_%s_v%s%s", nround12w, version12w, fileExtension12w) 

regressor12w = xgb.train(params = params,
                       data = Xmatrix12w, 
                       nround = nround12w,
                       early_stopping_rounds=25,
                       watchlist=list(val1=Xmatrix12w,val2=Xmatrix12w_t),
                       print_every_n = 25,
                       verbose = 1)

#Saves the regressor intelligence. Model can then be used without retraining algorithm.
# save the model to disk
saveRDS(regressor12w, "modelName12w")
turnover_pred12w = predict(regressor12w, Xmatrix12w_t)
test_result12 <- data.table(test_set12, turnover_pred12w)

#----------------------------------------------------------------------------------------------------------------------

test_result4$'Delta 4w [EUR]' <- round(test_result4$turnover_pred4w - test_result4$TREfuture4, 1)
test_result4$'Differences 4w [%]' <- round(ifelse(test_result4$'Delta 4w [EUR]' > 0, abs(test_result4$'Delta 4w [EUR]')/test_result4$TREfuture4*100, 
                                                   abs(test_result4$'Delta 4w [EUR]')/test_result4$turnover_pred4w*100),1)

test_result12$'Delta 12w [EUR]' <- round(test_result12$turnover_pred12w - test_result12$TREfuture12, 1)
test_result12$'Differences 12w [%]' <- round(ifelse(test_result12$'Delta 12w [EUR]' > 0, abs(test_result12$'Delta 12w [EUR]')/test_result12$TREfuture12*100, 
                                                   abs(test_result12$'Delta 12w [EUR]')/test_result12$turnover_pred12*100),1)
test_subject <- subset(test_result12, CustomerId == '30848')
#close the ODBC connection
dbDisconnect(DBI_Connection)

#Clears memory
rm(list = ls(all.names = TRUE))

