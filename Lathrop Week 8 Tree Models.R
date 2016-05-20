####################################################################################
# Andy Lathrop, PREDICT 422-DL_SEC59                                               #
# Fall 2015 | Dr. Wightman                                                         #
# Week 8 Programming Assignment : Tree Models                                 #
# An Introduction to Statistical Learning, with Applications in R (2013)           # 
# by G. James, D. Witten, T. Hastie, and R. Tibshirani. ISBN-13 978-1-4614-7137-0. #
# http://www-bcf.usc.edu/~gareth/ISL/                                              #
####################################################################################

library(tree)
library(ISLR)

#####
# Begin Chapter 8.3 Lab: Decision Trees
#####
# Fitting Classification Trees

data(Carseats)
summary(Carseats)

# Sales is a continuous varibale, so recode as binary variable for classification
High=ifelse(Carseats$Sales<=8,"No","Yes")

# use data.frame() function to merge High with rest of the data
Carseats=data.frame(Carseats,High)

# use tree() function to fit classification tree to predict High using all variables except Sales
tree.carseats=tree(High~.-Sales,Carseats)

# view training error rate
summary(tree.carseats)

plot(tree.carseats)

# The argument pretty = 0 instructs R to include the category names for any qualitative predictors, 
# rather than simply displaying a letter for each category
text(tree.carseats,pretty=0, all=TRUE, splits=TRUE)

# get branch info
tree.carseats

# create test set
set.seed(2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")

# correct predictions 71.5%, 57 incorrect predictions (nominal error rate)
table(tree.pred,High.test)
(86+57)/200

# try pruning (cross-validation)
set.seed(3)

# We use the argument FUN = prune.misclass in order to indicate that we want the 
# classification error rate to guide the cross-validation and pruning process, 
# rather than the default for the cv.tree() function, which is deviance.
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)

# k parameter retuned in cost-complexity, dev is nominal error rate (# mis-classified)
# tree with lowest error rate has 9 nodes
cv.carseats
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")

# tree with lowest error rate from cross-validation had 9 nodes
# so we will use this for pruning
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(94+60)/200 # now 77% correctly classified (46 incorrectly classified)

# test larger tree -> lower accuracy
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+62)/200
#####
# End Lab 8.3

# Begin Exercise 8 of Section 8.4
#####
# In the lab, a classification tree was applied to the Carseats data set 
# after converting Sales into a qualitative response variable. Now we will 
# seek to predict Sales using regression trees and related approaches, 
# treating the response as a quantitative variable.

####
# (a) Split the data set into a training set and a test set.
set.seed(93)
train = sample(1:nrow(Carseats), nrow(Carseats)/2)

####
# (b) Fit a regression tree to the training set. Plot the tree, and interpret the results.
# What test MSE do you obtain?
library(MASS)
set.seed(1)

carRegrTree =tree(Sales~. -High, Carseats, subset=train)

# Note: MSE = Residual Mean Deviance found in summary output
# deviance = sum of squared errors
summary(carRegrTree)

par(mfrow=c(1,1))
plot(carRegrTree)
text(carRegrTree,pretty=0, all=T, splits=T)

####
# (c) cross-validation
cvRegrTree=cv.tree(carRegrTree)
plot(cvRegrTree$size,cvRegrTree$dev,type='b')

# check exact values
cvRegrTree$size
cvRegrTree$dev

# check out pruned tree with 8 terminal nodes
pruneRegr=prune.tree(carRegrTree,best=8)

plot(pruneRegr)
text(pruneRegr,pretty=0, all=T)
summary(pruneRegr)

# code from lab that was not used in this exercise
# yhat=predict(tree.boston,newdata=Boston[-train,])
# boston.test=Boston[-train,"medv"]
# plot(yhat,boston.test)
# abline(0,1)
# mean((yhat-boston.test)^2)

####
# (d) Use the bagging approach in order to analyze this data. 
# What test MSE do you obtain? Use the importance() function 
# to determine which variables are most important.

# Bagging and Random Forests

library(randomForest)
set.seed(1)

####
# bagging model
# The argument mtry = 10 indicates that all 10 predictors should be considered 
# for each split of the tree — in other words, that bagging should be done.
carBag=randomForest(Sales~. -High, data=Carseats, subset=train, mtry=10, importance=TRUE)

# show model results
carBag

# calculate predictions on test set
yhat.bag = predict(carBag, newdata=Carseats[-train,])

# create vector of response (Sales) for test set for plotting
car.test <- Carseats[-train,"Sales"]

plot(yhat.bag, car.test)
abline(0,1)

# calc MSE
mean((yhat.bag-car.test)^2)

# calc importance
importance(carBag)
varImpPlot(carBag)

# this code not used (from textbook example) - reduces number of trees
# bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
# yhat.bag = predict(bag.boston,newdata=Boston[-train,])
# mean((yhat.bag-boston.test)^2)

####
# (e) Use random forests to analyze this data. What test MSE do you obtain? 
# Use the importance() function to determine which variables are most important. 
# Describe the effect of m, the number of variables considered at each split, 
# on the error rate obtained.

####
# random forest model
Carseats[,-which(names(Carseats) %in% c("Sales","High"))]

set.seed(1)
# use default setting of 'mtry' parameter for random forest, p/3
rfCar.default=randomForest(Sales~. -High ,data=Carseats, subset=train,importance=TRUE)
rfCar.default

set.seed(1)
# set mtry = 4 to see what happens
rfCar=randomForest(Sales~. -High ,data=Carseats, subset=train, mtry=4,importance=TRUE)
rfCar

set.seed(1)
# use the tuneRF function to search and plot different values of mtry
tuneRF(Carseats[,-which(names(Carseats) %in% c("Sales","High"))], Carseats[,"Sales"], 
       stepFactor=2, mtryStart=2, doBest=T, plot=T)


yhat.rf = predict(rfCar.default,newdata=Carseats[-train,])

# create vector of response (Sales) for test set for plotting
car.test <- Carseats[-train,"Sales"]

# calc test MSE
mean((yhat.rf-car.test)^2)

importance(rfCar.default)
varImpPlot(rfCar.default)
#####
# End Exercise 8 of Section 8.4

# Begin Exercise 9 of Section 8.4
#####
data(OJ)
summary(OJ)
plot(OJ$Purchase,OJ$PriceCH)

set.seed(1013)

# part (a)
# create training and test sets
train = sample(dim(OJ)[1], 800)
trainOJ = OJ[train, ]
testOJ = OJ[-train, ]

# part (b)
# fit tree
library(tree)
oj.tree = tree(Purchase ~ ., data = trainOJ)
summary(oj.tree)

# part c
# get detailed info
oj.tree

# part d
# plot
par(mfrow=c(1,1))

plot(oj.tree)
text(oj.tree, pretty = 0, all=T, splits=T)

# part e
# predict response on test data, produce confusion matrix
oj.pred = predict(oj.tree, testOJ, type = "class")
table(testOJ$Purchase, oj.pred)

# part f
# cross validation
cv.oj = cv.tree(oj.tree, FUN=prune.misclass)
cv.oj
par(mfrow=c(1,2))
plot(cv.oj$size,cv.oj$dev,type="b")
plot(cv.oj$k,cv.oj$dev,type="b")
par(mfrow=c(1,1))
plot(cv.oj)

# part g
# plot of cross-validation results
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

# part i
# produce pruned tree with optimal tree size
oj.pruned = prune.tree(oj.tree, best = 6)
summary(oj.pruned)
oj.pruned

par(mfrow=c(1,1))
plot(oj.pruned)
text(oj.pruned, pretty = 0, all=T, splits=T)

# park k
# compare  test error rates
# unpruned
pred.unpruned = predict(oj.tree, testOJ, type = "class")
misclass.unpruned = sum(testOJ$Purchase != pred.unpruned)
misclass.unpruned/length(pred.unpruned)
## [1] 0.1889

# pruned
pred.pruned = predict(oj.pruned, testOJ, type = "class")
misclass.pruned = sum(testOJ$Purchase != pred.pruned)
misclass.pruned/length(pred.pruned)
## [1] 0.1889
#####
# End Exercise 9 of Section 8.4

# Begin Exercise 10 of Section 8.4
# base code from
# https://github.com/asadoughi/stat-learning/blob/master/ch8/10.Rmd
# Use boosting to predict Salary in the Hitters data set
#####

# part (a)
# remove NAs for Salary, then log transform Salary
data(Hitters)
sum(is.na(Hitters$Salary))
Hitters = Hitters[-which(is.na(Hitters$Salary)), ]
sum(is.na(Hitters$Salary))
Hitters$Salary = log(Hitters$Salary)

# part (b)
# create training set of the first 200 observations, test set with the rest
train = 1:200
trnHitters = Hitters[train, ]
testHitters = Hitters[-train, ]

# part (c)
# boosting
library(gbm)
set.seed(93)
pows = seq(-10, -0.2, by=0.1)
lambdas = 10 ^ pows
length.lambdas = length(lambdas)
train.errors = rep(NA, length.lambdas)
test.errors = rep(NA, length.lambdas)
for (i in 1:length.lambdas) {
  boost.hitters = gbm(Salary~., data=trnHitters, distribution="gaussian", n.trees=1000, shrinkage=lambdas[i])
  train.pred = predict(boost.hitters, trnHitters, n.trees=1000)
  test.pred = predict(boost.hitters, testHitters, n.trees=1000)
  train.errors[i] = mean((trnHitters$Salary - train.pred)^2)
  test.errors[i] = mean((testHitters$Salary - test.pred)^2)
}

plot(lambdas, train.errors, type="b", xlab="Shrinkage", ylab="Train MSE", col="blue", pch=20)

# part (d)
plot(lambdas, test.errors, type="b", xlab="Shrinkage", ylab="Test MSE", col="red", pch=20)
min(test.errors)

# get lambda value for minimum test error
lambdas[which.min(test.errors)]

# get error value for optimal lambda
which.min(test.errors)
test.errors[which.min(test.errors)]

# part (e)
# compare booting to multiple linear regression and lasso
# multiple LR
lm.fit = lm(Salary~., data=trnHitters)
lm.pred = predict(lm.fit, testHitters)
mean((testHitters$Salary - lm.pred)^2)

# lasso
library(glmnet)
set.seed(94)
x = model.matrix(Salary~., data=trnHitters)
y = trnHitters$Salary
x.test = model.matrix(Salary~., data=testHitters)
lasso.fit = glmnet(x, y, alpha=1)
lasso.pred = predict(lasso.fit, s=0.01, newx=x.test)
mean((testHitters$Salary - lasso.pred)^2)

# Both linear model and regularization like Lasso have higher test MSE than boosting.

# part (f)
# variable importance
boost.best = gbm(Salary~., data=trnHitters, distribution="gaussian", n.trees=1000, 
                 shrinkage=lambdas[which.min(test.errors)])
summary(boost.best)

# CAtBat, CRBI and CWalks are three most important variables in that order

# part (g)
# bagging
library(randomForest)
set.seed(999)
rf.hitters = randomForest(Salary~., data=trnHitters, ntree=500, mtry=19)
rf.pred = predict(rf.hitters, testHitters)
mean((testHitters$Salary - rf.pred)^2)

# Test MSE for bagging is about $0.23$, which is slightly lower than the best test MSE for boosting.

#####
# End Exercise 10 of Section 8.4

