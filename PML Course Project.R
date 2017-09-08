# Pracitical Machine Learning - JHU
# Course Project  
############################################################################################
# BACKGROUND
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to
# collect a large amount of data about personal activity relatively inexpensively. These
# type of devices are part of the quantified self movement - a group of enthusiasts who
# take measurements about themselves regularly to improve their health, to find patterns
# in their behavior, or because they are tech geeks. One thing that people regularly do is
# quantify how much of a particular activity they do, but they rarely quantify how well they
# do it. In this project, your goal will be to use data from accelerometers on the belt,
# forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts
# correctly and incorrectly in 5 different ways. More information is available from the
# website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight
# Lifting Exercise Dataset).
#
# PROJECT GOAL
# The goal of your project is to predict the manner in which they did the exercise. This
# is the "classe" variable in the training set. You may use any of the other variables to
# predict with. You should create a report describing how you built your model, how you
# used cross validation, what you think the expected out of sample error is, and why you
# made the choices you did. You will also use your prediction model to predict 20 different
# test cases.
########################################################################################
setwd("~/R Projects Laptop/Practical Machine Learning - JHU")
library(readr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(ranger)
library(VIM)
set.seed(1010)

# Import Data
training <- read.csv("pml-training.csv", na.strings=c('#DIV/0!', '', 'NA') ,stringsAsFactors = F)
testing  <- read.csv("pml-testing.csv",  na.strings=c('#DIV/0!', '', 'NA') ,stringsAsFactors = F)

# Examine data structure
str(training)
## Convert variables to correct class
training$new_window <- as.factor(training$new_window)
training$kurtosis_yaw_belt <- as.numeric(training$kurtosis_yaw_belt)
training$skewness_yaw_belt <- as.numeric(training$skewness_yaw_belt)
training$kurtosis_yaw_dumbbell <- as.numeric(training$kurtosis_yaw_dumbbell)
training$skewness_yaw_dumbbell <- as.numeric(training$skewness_yaw_dumbbell)
qplot(training$cvtd_timestamp)
training$cvtd_timestamp  <- as.factor(training$cvtd_timestamp)
str(training)

## Repeat previous on testing data
testing$new_window <- as.factor(testing$new_window)
testing$kurtosis_yaw_belt <- as.numeric(testing$kurtosis_yaw_belt)
testing$skewness_yaw_belt <- as.numeric(testing$skewness_yaw_belt)
testing$kurtosis_yaw_dumbbell <- as.numeric(testing$kurtosis_yaw_dumbbell)
testing$skewness_yaw_dumbbell <- as.numeric(testing$skewness_yaw_dumbbell)
testing$cvtd_timestamp  <- as.factor(testing$cvtd_timestamp)

# Examine data sets
colnames(training)  # target variable is column 160 "classe"
colnames(testing)   # column 160 is "problem_id"
stripplot(training$X)  # Row numbers
stripplot(testing$X)   # Row numbers
stripchart(training$raw_timestamp_part_1)
stripchart(training$raw_timestamp_part_2)

# Survey Missing Values
aggr(training)

# How Much Missing Data
## Missing Values as fraction of total
sum(is.na(training))/(dim(training)[1]*dim(training)[2]) 

## Missing Values fraction by column
missCol <- apply(training, 2, function(x) sum(is.na(x)/length(x)))  

## Distribution of Missing Features
hist(missCol, main = "Missing Data by Column")
table(missCol)
missIndCol <- which(missCol > 0.9); length(missIndCol)  #Number of predictors > 90% missing

## Remove Missing Featues from training and test sets
train.xform.temp <- training[,-missIndCol]
test.xform.temp  <- testing[,-missIndCol]

## Remove X = row count variable, and raw time stamps
str(training)
train.xform  <- train.xform.temp[,-c(1,3,4)]
test.xform   <- test.xform.temp[,-c(1,3,4)]
str(train.xform)
aggr(train.xform)

## Examine Missing Cases;  All cases are complete
missRow <- apply(train.xform, 1, function(x) sum(is.na(x)/length(x)))
table(missRow)
sum(!complete.cases(train.xform))

## Near Zero Variance Features 
# nzv <- nearZeroVar(train.xform)
# length(nzv)

#######################################################################################
## Model used for Quiz;  90% Accuracy
# modNet = train(classe~., data = train.xform, method = "glmnet",
#                preProcess = c("nzv", "center", "scale"),
#                tuneGrid = expand.grid(alpha = 0:1, lambda = seq(.0001, 0.1, length = 100)),
#                trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
#                na.action = na.pass)
# 
# ## Model output
# modNet
# plot(modNet)
# predNet <- predict(modNet, train.xform)
# confusionMatrix(predNet, train.xform$classe)
# getTrainPerf(modNet)
# varImp(modNet)
# predNetTest <- predict(modNet, test.xform[,-92], na.action = na.pass)

#######################################################################################
## Fit a Random Forest
modFor  <- train(classe~., data = train.xform, method = "rf", trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE),
                 na.action = na.pass)

modFor
plot(modFor)
predFor <- predict(modFor, train.xform)
confusionMatrix(predFor, train.xform$classe)
getTrainPerf(modFor)
varImp(modFor)
predForTest <- predict(modFor, test.xform[,-92], na.action = na.pass)
predForTest

