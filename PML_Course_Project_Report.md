# Practical Machine Learning - Course Project
Mark Blackmore  
July 26, 2017  


# Introduction
This is the course project write up for the Johns Hopkins University Data Science Specialization's course on Practical Machine Leaning.  The project involves reading in training and test data sets, and creating a machine learning model for prediction.  This document outlines the steps taken to produce the model and to make predictions.

# Project Description
## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to
collect a large amount of data about personal activity relatively inexpensively. These
type of devices are part of the quantified self movement - a group of enthusiasts who
take measurements about themselves regularly to improve their health, to find patterns
in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Project Goal
The goal of your project is to predict the manner in which they did the exercise. This
is the "classe" variable in the training set. You may use any of the other variables to
predict with. You should create a report describing how you built your model, how you
used cross validation, what you think the expected out of sample error is, and why you
made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Data
Training data can be found here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Test data can be found here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for the project came from this source
http://groupware.les.inf.puc-rio.br/har

# Data Wrangling
## R Environment; Data Import


```r
# Set up environment
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(ranger)
library(VIM)
set.seed(1010)

# Import Data
training <- read.csv("pml-training.csv", na.strings=c('#DIV/0!', '', 'NA'), stringsAsFactors = F)
testing  <- read.csv("pml-testing.csv",  na.strings=c('#DIV/0!', '', 'NA'), stringsAsFactors = F)
```

## Training data: change variable class

```r
# str(training)
# Convert training variables to apporiate class
training$new_window <- as.factor(training$new_window)
training$kurtosis_yaw_belt <- as.numeric(training$kurtosis_yaw_belt)
training$skewness_yaw_belt <- as.numeric(training$skewness_yaw_belt)
training$kurtosis_yaw_dumbbell <- as.numeric(training$kurtosis_yaw_dumbbell)
training$skewness_yaw_dumbbell <- as.numeric(training$skewness_yaw_dumbbell)
# qplot(training$cvtd_timestamp)  # should be a factor variable
training$cvtd_timestamp  <- as.factor(training$cvtd_timestamp)
```
Sevral skewness and kurtosis variables were coerced from class factor to class numeric. A timestamp variable was coerced to factor based on the observation that stamps were categorical in nature.

## Testing data: change variable class

```r
# Convert testing variables to arropriate class
testing$new_window <- as.factor(testing$new_window)
testing$kurtosis_yaw_belt <- as.numeric(testing$kurtosis_yaw_belt)
testing$skewness_yaw_belt <- as.numeric(testing$skewness_yaw_belt)
testing$kurtosis_yaw_dumbbell <- as.numeric(testing$kurtosis_yaw_dumbbell)
testing$skewness_yaw_dumbbell <- as.numeric(testing$skewness_yaw_dumbbell)
testing$cvtd_timestamp  <- as.factor(testing$cvtd_timestamp)
```
Class conversion on the training set were replicated on the test set.

# Missing Values
## Plot Missing Values

```r
aggr(training)
```

![](PML_Course_Project_Report_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

The plot shows several variables with high proportions of missing data, with some variables nearly missing entirely.

## How much data is missing?

```r
# Missing Values as fraction of total
sum(is.na(training))/(dim(training)[1]*dim(training)[2]) 
```

```
## [1] 0.6131835
```

```r
# Missing Values fraction by column / variable
missCol <- apply(training, 2, function(x) sum(is.na(x)/length(x)))  

# Distribution of Missing Variables
hist(missCol, main = "Missing Data by Variable")
```

![](PML_Course_Project_Report_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
# table(missCol)
missIndCol <- which(missCol > 0.9); length(missIndCol)  #Number of predictors > 90% missing
```

```
## [1] 100
```
Sixty one percent of the total or full data array are missing. One hundred variables had in excess of ninety percent missing data.  We removed these latter variables and unneccesary observations such as row nummbers and raw timestamps.  

## Remove variables

```r
# Remove Missing Variables from training and test sets
train.xform.temp <- training[,-missIndCol]
test.xform.temp  <- testing[, -missIndCol]

# Remove X = row count variable, and raw time stamps
train.xform  <- train.xform.temp[,-c(1,3,4)]
test.xform   <- test.xform.temp[,-c(1,3,4)]
aggr(train.xform)
```

![](PML_Course_Project_Report_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

## Complete Cases

```r
# Examine Missing Cases;  All cases are complete
sum(!complete.cases(train.xform))
```

```
## [1] 0
```
We verified that all remiaing cases are complete, that there is no missing data.

# Machine Learning Model 
## Random Forest Model under 10-fold Cross Validation

```r
# Fit a Random Forest
modFor  <- train(classe~., data = train.xform, method = "rf", trControl = trainControl(method = "cv", number = 10, verboseIter = FALSE), na.action = na.pass)
```
We fit a Random Forest machine learning model.  We used the entire training set and 10-fold cross-validation to find the hyperparameter "mtry" for number of variables for splitting at each node. We used default values for "mtry."

## Model Evaluation

```r
modFor
```

```
## Random Forest 
## 
## 19622 samples
##    56 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17661, 17660, 17660, 17659, 17658, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9880236  0.9848475
##   40    0.9987768  0.9984529
##   78    0.9981144  0.9976148
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 40.
```

```r
getTrainPerf(modFor)
```

```
##   TrainAccuracy TrainKappa method
## 1     0.9987768  0.9984529     rf
```
Cross validated accuracy is nearly 100%; out of sample error is less than 0.2%  We saw no reason to seek a better model.  We  anticipate a reduction in accuracy when the data is applied to our testing obervations.

## Prediction 

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
Predictions based on course quiz submission were 100% correct.

# Conclusion
Data wrangling and treating missing observation was essential to this analysis.  The result of these actions led to a training set of complete cases.  No further pre-processing was required. A Random Forest model fit to this training data using default parameters produced 100% accuracy in prediction, an unexpectedly positive result.
