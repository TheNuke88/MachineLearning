## Client value prediction 
## 
#Clears memory
rm(list = ls(all.names = TRUE))
# Preample

#Install commands
#install.packages("devtools", dependencies = TRUE)
#install.packages("odbc", dependencies = TRUE)
#install.packages("DBI", dependencies = TRUE)
#install.packages("data.table", dependencies = TRUE)
#install.packages("caTools", dependencies = TRUE)
#install.packages("xtable", dependencies = TRUE)
#install.packages("dplyr", dependencies = TRUE)
#install.packages("formattable", dependencies = TRUE)
#install.packages("RcppRoll", dependencies = TRUE)
#install.packages("caret", dependencies = TRUE)
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

DBI_Connection <- dbConnect(odbc(), .connection_string = "Driver={SQL Server};server=<insert server name>;UID=;PWD=;database=", timeout = 30)
DBI_Connection
# Querying full dataset
#--------------------------------------------------------------------------------------------------------
dt_fullData <- dbFetch(dbSendQuery(DBI_Connection, 
"CLIENT DATA GOES HERE"))

dt_Dates <- dbFetch(dbSendQuery(DBI_Connection, 
"DATE QUERY GOES HERE"))
dt_fullData$StartWeekDate <- as.character(dt_fullData$StartWeekDate)
dt_Dates$Date <- as.character(dt_Dates$Date)
dt_fullData_ <- merge(dt_fullData, dt_Dates, by.x = "StartWeekDate", by.y = "Date", all.x = TRUE)

dt_fullData <- subset(dt_fullData_, select = -c(StartWeekDate, EndWeekDate, Month.x))
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
# Forecasting 4 weeks 
#---------------------------------------------------------------------------------------------------------
nround4w = 851  #number rounds of ensemble
version4w = 2   #version number of model used
fileExtension4w = ".dat"
modelName4w = sprintf("xgboost4w_%s_v%s%s", nround4w, version4w, fileExtension4w) 

regressor4w = readRDS("modelName4w") #reads a previously trained regressor 

#---------------------------------------------------------------------------------------------------------
# Forecasting 12 weeks
#---------------------------------------------------------------------------------------------------------
nround12w = 851 #number rounds of ensemble
version12w = 2   #version number of model used
fileExtension12w = ".dat"
modelName12w = sprintf("xgboost12w_%s_v%s%s", nround12w, version12w, fileExtension12w) 

regressor12w = readRDS("modelName12w") #reads a previously trained regressor 

#----------------------------------------------------------------------------------------------------------------------
# Saving important bits to vector and then dumping to database
#----------------------------------------------------------------------------------------------------------------------
dt_2018Data <- dt_fullData[(dt_fullData$Year > 2018),]


dt_2018Data4 <- na.omit(dt_2018Data)
dt_2018Data12 <- na.omit(dt_2018Data)

dt_2018DataX4 <- model.matrix(~ . -1, data = dt_2018Data4)
Xmatrix4 <- xgb.DMatrix(data = dt_2018DataX4)

dt_2018DataX12 <- model.matrix(~ . -1, data = dt_2018Data12)
Xmatrix12 <- xgb.DMatrix(data = dt_2018DataX12)

CVS_4weeks = predict(regressor4w, newdata = Xmatrix4)
CVS_12weeks = predict(regressor12w, newdata = Xmatrix12)

uncertainties_subset <-  data.table(dt_2018Data4, ifelse(CVS_4weeks < 0, 0, CVS_4weeks) , ifelse(CVS_4weeks > CVS_12weeks, CVS_4weeks, CVS_12weeks))

#-----------------------------------------------------------------------------------------------------------------------
# Estimation of an error on the prediction - using STD of the predictions
#-----------------------------------------------------------------------------------------------------------------------


dt_uncertainties <- uncertainties_subset %>% 
  group_by(CustomerId) %>%
  mutate( meanForecast4 = roll_meanr(V2, 3, fill = 0),
          meanForecast12 = roll_meanr(V3, 3, fill = 0),
          )
dt_latestData <- dt_uncertainties[(dt_uncertainties$Year > 2018),]
# ------------------------------------------------------------------------------------------------------------------------------------------------


CVS_scoreVector <-  data.table(dt_latestData$CustomerId,
                          dt_latestData$Week,
                          dt_latestData$Year,
                          dt_latestData$customerWeeklyTurnover,
                          round(ifelse(dt_latestData$V2 < 0, 0, dt_latestData$V2),1), 
                          round(abs(dt_latestData$V2 - dt_latestData$meanForecast4),1), 
                          round(ifelse(dt_latestData$V2 > dt_latestData$V3, dt_latestData$V2, dt_latestData$V3),1), 
                          round(abs(dt_latestData$V3-dt_latestData$meanForecast12),1),
                          modelName4w, 
                          modelName12w, 
                          Sys.time()
                          )


setnames(CVS_scoreVector,
         c("V1", "V2", "V3", "V4", "V5", "V6" , "V7", "V8" ,"modelName4w" , "modelName12w" ,"V11"),
         c("CustomerID",
           "Week",
           "Year",
           "WeeklyTurnover [EUR]",
           "next4Weekforecast",
           "Uncertainty4week",
           "next12Weekforecast",
           "Uncertainty12week",
           "Model_4weeks",
           "Model_12weeks",
           "CVS_creationDate")
         )
CVS_scoreVector$CustomerID = as.numeric(CVS_scoreVector$CustomerID)

# last week's data
# CVS_scoreVector <- CVS_scoreVector[(CVS_scoreVector$Week == week(Sys.Date())) & (CVS_scoreVector$Year == year(Sys.Date())),]



#write local filen
#currentDate = Sys.Date()
#fileExtensionCSV = ".csv"
#filename = sprintf("CVS_scoreVector_v%s%s",  currentDate, fileExtensionCSV)
#write.csv(CVS_scoreVector,filename)

#Connect to DB
dbWriteTable(DBI_Connection,'CustomerValueScore', CVS_scoreVector, append = TRUE)
#close the ODBC connection
dbDisconnect(DBI_Connection)

#Clears memory
rm(list = ls(all.names = TRUE))

