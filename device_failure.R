# Company (3D Technologies) has a fleet of devices transmitting daily aggregated telemetry
# attributes.
# Predictive maintenance techniques are designed to help determine the condition of in-service
# equipment in order to predict when maintenance should be performed. This approach
# promises cost savings over routine or time-based preventive maintenance, because tasks are
# performed only when warranted.
# Goal
# You are tasked with building a predictive model using machine learning to predict the
# probability of a device failure. When building this model, be sure to minimize false positives and
# false negatives. The column you are trying to predict is called failure with binary value 0 for
# non-failure and 1 for failure.

setwd("C:\\Users\\SROY\\Documents\\CodeBase\\Datasets\\aws")
rm(list=ls())
seedVal = 17869
options(warn=-1)

#Load libraries
library(parallel)
library(doMC)
library(DMwR)
library(zoo)
library(ROSE)
library(mlr)
library(dplyr)
library(pROC)
library(gridExtra)
library(caret)
library(xgboost)
library(ggplot2)
library(lubridate)
library(corrplot)
library(reshape2)
library(dummies)
library(h2o)

theme_set(theme_classic())
set.seed(seedVal)

#Import Data
df <- read.csv2('device_failure_aws.csv', sep = ',', header = TRUE)

# Check structure of data
str(df)
#Fix Datatype of failure
df$failure <- as.factor(df$failure)

# Date looks weird. Case study says YYYY-MM-DD format. Surly it's not in the same format.
length(unique(df$date)) # 304 elements
# Need to check if there is a pattern
length(grep('15*', unique(df$date))) # 304
# Looks like the year is 2015 and the rest are the days. Fix it using lubridate.
df$date <- substr(df$date, start = 3, stop = 5)
df$date <- as_date(as.integer(df$date), origin = "2014-12-31")
# create extra date value columns
df$wday <- lubridate::wday(df$date)
df$week <- lubridate::week(df$date)
df$month <- lubridate::month(df$date)
df <- df %>% select(date, wday, week, month, device, starts_with('att'), failure)
# Convert them into factors
df <- df %>% mutate_at(vars(wday, week, month), funs(factor(.)))

# Check missing values
apply(df, 2, function(x) length(which(x == "" | is.na(x) | x == "NA"))) # No missing values

# Devices
length(unique(df$device)) #1168 unique devices
# Number of devices recorded each day.
countDeviceEachDay <- df %>% select(date) %>% count(date)
ggplot(countDeviceEachDay, aes(x=date)) + geom_line(aes(y=n)) + labs(x='Time', y='Devices Recorded')
# Device recordings are decreasing

# Let's check the date range of all devices
countDeviceDateRange <- df %>% select(date, device) %>% group_by(device) %>% 
  mutate(maxDate = max(date)) %>% mutate(minDate = min(date)) %>%
  select(-date) %>% distinct() %>% mutate(runningDays = (ymd(maxDate) - ymd(minDate))[[1]]) %>%
  arrange(desc(runningDays))
# 31 devices are still running.
# So number of failures should be = devices taken off
nrow(df[df$failure == 1,]) #106. Why? Were the devices fixed and reintroduced?

# Let's check the above discrepancy
countDeviceRestarts <- df %>% select(date, device, failure) %>% group_by(device) %>% 
  mutate(maxDate = max(date)) %>%
  mutate(failureDate = if_else(failure == 1, date, NULL)) %>% select(-date) %>% distinct()
dupDevice <- countDeviceRestarts %>% select(device) %>% count() %>% filter(n > 1)
#S1F023H2
# At this point it looks clear that many devices were taken off without failure
countDeviceRestarts <- countDeviceRestarts %>%
  mutate(Status = if_else(is.na(failureDate), 'Terminated-Unknown', 
                          if_else(failureDate == maxDate, 'Terminated-Failure', 'Restarted')))
countDeviceRestarts[countDeviceRestarts$device %in% dupDevice$device & countDeviceRestarts$failure==0,'Status'] <- 'Duplicated'
countDeviceRestarts <- countDeviceRestarts[countDeviceRestarts$Status != 'Duplicated',]
countDeviceRestarts[countDeviceRestarts$maxDate == '2015-11-02', 'Status'] <- 'Still Monitored'

# From above it is observed that some devices were restarted
View(countDeviceRestarts[countDeviceRestarts$Status == 'Restarted',])
freqtable <- table(countDeviceRestarts$Status)
freqtable <- as.data.frame.table(freqtable)
ggplot(freqtable, aes(Var1, Freq)) + geom_bar(stat="identity", width = 0.5, fill="tomato2") + 
  labs(x = "Device Status", y = "Count of devices") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

# Check if there are more than 1 failures. No devices failed
countManyDeviceFailures <- df %>% select(device, failure) %>% filter(failure==1) %>%
  select(device) %>% group_by(device) %>% count() %>% filter(n > 1)

# Devices which started late.
View(countDeviceDateRange[countDeviceDateRange$minDate > '2015-01-01',])

# Average uptime in days
summary(countDeviceDateRange$runningDays)
nrow(countDeviceDateRange[countDeviceDateRange$runningDays <= 5 ,])
countDeviceDateRange$uptime <- cut(countDeviceDateRange$runningDays, 
                                   breaks = c(-Inf, 5, 85, 225, 300, Inf), 
                                   labels = c("Extremely Low (0-5)","Low (5-85)","Medium (86-225)","High (226-300)", "StillUp"))
freqtable <- table(countDeviceDateRange$uptime)
freqtable <- as.data.frame.table(freqtable)
ggplot(freqtable, aes(Var1, Freq)) + geom_bar(stat="identity", width = 0.5, fill="tomato2") + 
  labs(x = "Uptime category (days)", y = "Count of devices") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

# Lets count start date and end date
countDeviceEndDate <- countDeviceDateRange %>% group_by(maxDate) %>% mutate(count = n()) %>% 
  select(maxDate, count) %>% distinct()
countDeviceStartDate <- countDeviceDateRange %>% group_by(minDate) %>% mutate(count = n()) %>% 
  select(minDate, count) %>% distinct()

# Lets analyze device termination days
countDeviceEndDate$month <- lubridate::month(countDeviceEndDate$maxDate, label = TRUE)
countDeviceEndDate$wday <- lubridate::wday(countDeviceEndDate$maxDate, label = TRUE)
countDeviceEndDate$week <- lubridate::week(countDeviceEndDate$maxDate)
countDeviceEndDate <- countDeviceEndDate %>% arrange(desc(count))
head(countDeviceEndDate)

# Count per month
countTermPerMonth <- countDeviceEndDate %>% select(month, count) %>% 
  group_by(month) %>% summarise(Terminations = sum(count))
ggplot(countTermPerMonth[-nrow(countTermPerMonth),], aes(month, Terminations)) + 
  geom_bar(stat="identity", width = 0.5, fill="tomato3") + 
  labs(x = "Termination Month", y = "Count of devices") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

# Count per week
countTermPerWeek <- countDeviceEndDate %>% select(week, count) %>% 
  group_by(week) %>% summarise(Terminations = sum(count))
ggplot(countTermPerWeek, aes(x=week)) + geom_line(aes(y=Terminations, col='red')) + 
  labs(x='Termination Week Number', y='Count of devices')

# Count per week day
countTermPerWday <- countDeviceEndDate %>% select(wday, count) %>% 
  group_by(wday) %>% summarise(Terminations = sum(count))
ggplot(countTermPerWday, aes(wday, Terminations)) + 
  geom_bar(stat="identity", width = 0.5, fill="tomato4") + 
  labs(x = "Termination Week Day", y = "Count of devices") +
  theme(axis.text.x = element_text(angle=65, vjust=0.6))

# Correlation between attributes
corMatrix <- cor(df[,6:14])
corrplot(round(corMatrix,2), title = "Correlation between attributes", diag = FALSE)
# Attribute 7 and 8 are correlated
corMatrix[lower.tri(corMatrix,diag=TRUE)]=NA
corMatrix=as.data.frame(as.table(corMatrix))
corMatrix=na.omit(corMatrix)
corMatrix=corMatrix[order(-abs(corMatrix$Freq)),]
corMatrix[1:3,]

# Check attributes
# Plot density
ggplot(df, aes(attribute1)) + geom_density(aes(fill=failure), alpha=0.8)
ggplot(df, aes(attribute5)) + geom_density(aes(fill=failure), alpha=0.8) 
#Others are highly skewed
summary(df[,6:14])
# Really sparse. But will not remove outliers since they might infuence success/failure
summary(df[df$failure==1,6:14])
summary(df[df$failure==0,6:14])
# The mean changes a lot for attribute 2, 4 and 8 for failure/non-failure


# Attribute pattern before failure
failedDevs <- countDeviceRestarts[countDeviceRestarts$Status == 'Terminated-Failure' ,'device']
attDeviceFailure <- df %>% filter(device %in% failedDevs$device[1:20]) %>% arrange(device, date)
g1 <- ggplot(attDeviceFailure, aes(x=date)) + geom_line(aes(y=attribute6, col=device))
g2 <- ggplot(attDeviceFailure, aes(x=date)) + geom_line(aes(y=attribute4, col=device))
grid.arrange(g1, g2, nrow = 1)
# We can notice attribute 6 increases till failure. Is this true for non failure?
nonfailedDevs <- countDeviceRestarts[countDeviceRestarts$Status == 'Terminated-Unknown' ,'device']
attDevicenonFailure <- df %>% filter(device %in% nonfailedDevs$device[1:20]) %>% arrange(device, date)
ggplot(attDevicenonFailure, aes(x=date)) + geom_line(aes(y=attribute6, col=device))
# Not that much. We may say that increase in attribute 6 may lead to failure

# Let's check near zero variance to handle sparse columns. 
nearZeroVar(df[,-ncol(df)], names = TRUE)

# Device type analysis
df$devType <- substr(df$device,1,3)
unique(df$devType)
ggplot(df, aes(devType)) + geom_bar(aes(fill=failure), width = 0.5)
# devices that have failed
prop.table(table(df[df$failure==1,'devType']))

# # Since we are detecting failures and from above example, we should remove Terminated-Unknown
# df1 <- df %>% filter(!device %in% nonfailedDevs$device)
# df1 <- df1 %>% mutate_at(vars(starts_with('att')), funs(scale(.)))

# Class balance
prop.table(table(df$failure))
# Class is highly imbalanced

# Prepare Data
# Modelling wday and month didn't work great
#originalDf <- df
df <- originalDf
df$week <- as.integer(as.character(df$week))
lowrundevices <- countDeviceDateRange %>% filter(runningDays > 5)
df <- df %>% filter(device %in% lowrundevices$device)
#df <- df %>% filter(!device %in% nonfailedDevs$device)
# Adding device type to the model did not help.

# Create features. 
# We can apply rolling mean to all attributes. short and long
# We can apply sd to attributes not identified by near zero variance

# Rolling mean for columns 1,5,6,9
df <- df %>%
  group_by(device) %>%
  arrange(device, date) %>%
  mutate(rollmean_s_att1 = rollmean(x = attribute1, 5, align = "right", fill = 0),
         #rollmean_s_att5 = rollmean(x = attribute5, 5, align = "right", fill = 0),
         rollmean_s_att6 = rollmean(x = attribute6, 5, align = "right", fill = 0))
#rollmean_s_att9 = rollmean(x = attribute9, 5, align = "right", fill = 0))


# Roll sd for columns 1,5,6,9
df <- df %>%
  group_by(device) %>%
  arrange(device, date) %>%
  mutate(roll_sd_att1 = rollapply(attribute1, width = 5, FUN = sd, fill = 0, align = 'r'),
         #roll_sd_att5 = rollapply(attribute5, width = 5, FUN = sd, fill = 0, align = 'r'),
         roll_sd_att6 = rollapply(attribute6, width = 5, FUN = sd, fill = 0, align = 'r'))
#roll_sd_att9 = rollapply(attribute9, width = 5, FUN = sd, fill = 0, align = 'r'))

# # Drift for all columns
# drift <- function(att, n) {rev(att)[1:min(length(att), n)] %>% rev()}
# df <- df %>%
#   group_by(device) %>%
#   arrange(device, date) %>% 
#   summarise(drift_att1 = attribute1[1] - drift(attribute1, 1),
#             drift_att2 = attribute1[1] - drift(attribute2, 1),
#             drift_att3 = attribute1[1] - drift(attribute3, 1),
#             drift_att4 = attribute1[1] - drift(attribute4, 1),
#             drift_att5 = attribute1[1] - drift(attribute5, 1),
#             drift_att6 = attribute1[1] - drift(attribute6, 1),
#             drift_att8 = attribute1[1] - drift(attribute7, 1),
#             drift_att9 = attribute1[1] - drift(attribute8, 1))


# Select relevant columns
df <- df %>% select(-c(date, device, month, wday, devType, attribute7)) 
df <- df[,-1]
df <- df[,c(names(df)[!names(df) %in% 'failure'], 'failure')]
levels(df$failure) <- c("class0", "class1")

# Stratefied Sampling
apply(df, 2, function(x) length(which(is.na(x))))
inTrain <- createDataPartition(y = df$failure, list = FALSE, p = .8)
train <- df[inTrain,]
test <- df[-inTrain,]
stopifnot(nrow(train) + nrow(test) == nrow(df))
stopifnot(levels(test$failure) > 1)

############# Fit AutoML ##########

# Unbalanced class can be a problem as the feature is still not there
# But tree based algorithm shouldn't care much

h2o.init(nthreads=-1)

y <- "failure"
x <- setdiff(names(train), y)
train_smote <- SMOTE(failure ~ ., as.data.frame(train), perc.over = 30000)
prop.table(table(train_smote$failure))

aml <- h2o.automl(x = x, y = y,
                  training_frame = as.h2o(train_smote),
                  leaderboard_frame = as.h2o(test),
                  stopping_metric = 'AUC',
                  max_runtime_secs = 900,
                  seed = seedVal)

aml@leaderboard

aml@leader
pred <- h2o.predict(aml@leader, test)

##############################################

# Grid Tuning
registerDoMC(cores = 4)
# Max shrinkage for gbm
nl = nrow(train)
max(0.01, 0.1*min(1, nl/10000))

# Max Value for interaction.depth
floor(sqrt(NCOL(train)))

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 6),
                        n.trees = (0:50)*5, 
                        shrinkage = c(.1, .05,.025),
                        n.minobsinnode = c(5, 10, 15, 20))

# Fit gbm
trctrl <- trainControl(method = "repeatedcv", 
                       number = 3, 
                       repeats = 2, 
                       verboseIter = TRUE,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary,
                       sampling = "smote")

model.gbm <- caret::train(failure ~ . 
                          , data = train
                          , method = "gbm"
                          , metric = "ROC"
                          , preProcess = c("scale", "center")
                          , na.action = na.omit
                          , tuneGrid = gbmGrid
                          , trControl = trctrl)
model.gbm
gbmImp <- varImp(model.gbm, scale = FALSE)
gbmImp
plot(gbmImp)

# Plotting the model.
plot(model.gbm)

# Predict
pred <- predict(model.gbm, test[,-ncol(test)], type = "raw")
predicted <- ifelse(pred == 'class0', 0, 1)

# score prediction using AUC
reference <- ifelse(test$failure == 'class0', 0, 1)
auc <- roc(reference, predicted)
print(auc)
confusionMatrix(reference = reference, data = predicted)