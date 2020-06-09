#Haritha Ramachandran

library(randomForest)
library(caret)
library(e1071)
library(ROCR)
library(adabag)
library(rpart)
library(nnet)
library(kernlab)

set.seed(192)

solar <- read.csv("data_project_deepsolar.csv")
dim(solar)

#check for important variables
fitrf <-randomForest(solar_system_count~.,data =solar,importance =TRUE)
varImpPlot(fitrf, type=1)
sol <- solar[,c("solar_system_count","land_area","total_area","population_density","average_household_size","employed","average_household_income","population","housing_unit_median_value","household_count","occupancy_vacant_rate","heating_fuel_coal_coke_rate","housing_unit_count","race_white_rate","poverty_family_below_poverty_level_rate","education_bachelor_rate","race_asian_rate","heating_fuel_gas_rate","education_high_school_graduate_rate","education_less_than_high_school_rate","unemployed","race_other_rate","voting_2016_gop_percentage","heating_fuel_electricity_rate","number_of_years_of_education","race_black_africa_rate","per_capita_income","voting_2012_gop_percentage","voting_2016_dem_percentage","voting_2012_dem_percentage","occupation_finance_rate")]
str(sol)
#hence cateorical variabless are not among important ones and hence we can remove them

#remove categorical variables
s <- solar$solar_system_count
solar <- solar[, sapply(solar, is.numeric)]
dim(solar)

#compute correlation and remove highly correlated variables
c<-cor(solar)
highlyCorrelated <- findCorrelation(c, cutoff=0.9)  
solar<-solar[,-highlyCorrelated]  
dim(solar)  
solar<- cbind(s,solar)
dim(solar) 
colnames(solar)[1] <- "solar_system_count"


#select important variables 
fitrf2 <-randomForest(solar_system_count~.,data =solar,importance =TRUE)
varImpPlot(fitrf2, type=1, main="important variables")
solar <- solar[,c("solar_system_count","total_area","population_density","employed","average_household_income","voting_2012_gop_percentage","household_count","housing_unit_median_value","average_household_size","occupancy_vacant_rate","heating_fuel_gas_rate","heating_fuel_coal_coke_rate","unemployed","occupation_finance_rate","poverty_family_below_poverty_level_rate","education_less_than_high_school_rate","number_of_years_of_education","education_bachelor_rate","occupation_agriculture_rate","race_other_rate","education_college_rate","heating_fuel_electricity_rate","race_asian_rate","heating_fuel_none_rate","race_white_rate","employ_rate","gini_index","occupation_administrative_rate","education_professional_school_rate","occupation_construction_rate","education_high_school_graduate_rate")]
dim(solar)  


#split into train test 
N <-nrow(solar)
train <-sample(1:N,size =0.50*N)
test <-setdiff(1:N,train)

train_data <- solar[train,] #training data
test_data <- solar[test,] #testing data


#fit the logistic model
fit<- glm( solar_system_count~.,  data=train_data , family = "binomial")
summary(fit)

tau <-0.5   #initial value for tau
p <-fitted(fit)
pred <-ifelse(p>tau,"high","low")  
# cross tabulation between observed and predicted
tab<-table(train_data$solar_system_count, pred)
tab
#roc curve
predObj <-prediction(fitted(fit), train_data$solar_system_count)
perf <-performance(predObj,"tpr","fpr")
plot(perf)
abline(0,1,col ="darkorange2",lty =2)
# compute the area under the ROC curve
auc <-performance(predObj,"auc")
auc@y.values 



table(train_data$solar_system_count) #unbalanced  data 
sens <-performance(predObj,"sens")
spec <-performance(predObj,"spec")
tau <-sens@x.values[[1]]
sensSpec <-sens@y.values[[1]]+spec@y.values[[1]]
best <-which.max(sensSpec)
plot(tau, sensSpec,type ="l")
points(tau[best], sensSpec[best],pch =19,col =adjustcolor("red",0.5))
tau[best]

#classification for optimal tau
pred2 <-ifelse(fitted(fit)>tau[best],"high","low")
tab2<-table(train_data$solar_system_count, pred2)
tab2



#random forest
fit_rf <-randomForest(solar_system_count~.,data =train_data,importance =TRUE)
predRf <- predict(fit_rf, type = "response", newdata = test_data)
tabRf <- table(test_data$solar_system_count, predRf)
acc <- sum(diag(tabRf))/sum(tabRf)
acc 

predRf2 <- predict(fit_rf, type = "response", newdata = train_data)
tabRf2 <- table(train_data$solar_system_count, predRf2)
acc2 <- sum(diag(tabRf2))/sum(tabRf2)
acc2



#class tree
fitCt <-rpart(solar_system_count~.,data =train_data)
class <-predict(fitCt, type="class")
head(class)
table(class, train_data$solar_system_count)
phat <-predict(fitCt)
head(phat)

predObj <-prediction(phat[,2], train_data$solar_system_count)
# phat[,2] contains probability of class "low"
roc <-performance(predObj,"tpr","fpr")
plot(roc, main="ROC for train data")
abline(0,1,col ="darkorange2",lty =2)
auc <-performance(predObj,"auc")
auc@y.values 

phat2<- predict(fitCt, newdata= test_data)
head(phat2)
class2 <-predict(fitCt, type="class",newdata= test_data)
head(class2)
table(class2, train_data$solar_system_count)
predObj2 <-prediction(phat2[,2], test_data$solar_system_count)
# phat[,2] contains probability of class "yes"
roc <-performance(predObj2,"tpr","fpr")
plot(roc, main= "ROC for test data")
abline(0,1,col ="darkorange2",lty =2)
auc <-performance(predObj,"auc")
auc@y.values 


#svm
fitsvm <-ksvm(solar_system_count~.,data =train_data)
predValsvm <- predict(fitsvm, type = "response", newdata = test_data)
tabValsvm <- table(test_data$solar_system_count, predValsvm)
acc_svm <- sum(diag(tabValsvm))/sum(tabValsvm)
acc_svm


predValsvm2 <- predict(fitsvm, type = "response", newdata = train_data)
tabValsvm2 <- table(train_data$solar_system_count, predValsvm2)
acc_svm2<- sum(diag(tabValsvm2))/sum(tabValsvm2)
acc_svm2


#bagging
fitbag <-bagging(solar_system_count~.,data =train_data)

predValBag <-predict(fitbag,type ="class",newdata =train_data)
tabValBag <- table(train_data$solar_system_count, predValBag$class)
sum(diag(tabValBag))/sum(tabValBag)

predValBag2 <-predict(fitbag,type ="class",newdata =test_data)
tabValBag2 <- table(test_data$solar_system_count, predValBag2$class)
sum(diag(tabValBag2))/sum(tabValBag2)


# replicate the process a number of times
R <- 50
out <- matrix(NA, R, 7)
colnames(out) <- c("val_logistic", "val_randomforest","val_classification_tree","val_svm","val_bagging", "best", "test accuracy")
out <- as.data.frame(out)

for ( r in 1:R ) {
  
  N <-nrow(solar)
  train <-sample(1:N,size =0.50*N)
  val <-sample(setdiff(1:N, train),size =0.25*N )
  test <-setdiff(1:N,union(train, val))
  
   train_data <- solar[train,] #training data
  val_data <- solar[val,]  #validation data
  test_data <- solar[test,] #testing data
  
  
  
  # fit classifiers to only the training data
  fitLog<- glm( solar_system_count~. , data=train_data , family = "binomial")
  fitrf <-randomForest(solar_system_count~.,data =train_data,importance =TRUE)
  fitCt <-rpart(solar_system_count~.,data =train_data)
  fitsvm <-ksvm(solar_system_count~.,data =train_data)
  fitbag <-bagging(solar_system_count~.,data =train_data)
  
  # classify the validation data observations
  
  pred2 <-predict(fitLog,type ="response",newdata =val_data)# logistic regression
  predValLog <-ifelse(pred2>0.5,"high","low")
  tabValLog <- table(val_data$solar_system_count, predValLog)
  acc_log <- sum(diag(tabValLog))/sum(tabValLog)
  
  
  predValRf <- predict(fitrf, type = "response", newdata = val_data)
  tabValRf <- table(val_data$solar_system_count, predValRf)
  acc_rf <- sum(diag(tabValRf))/sum(tabValRf)
  
  predValCt <- predict(fitCt, type = "class", newdata = val_data)
  tabValCt <- table(val_data$solar_system_count, predValCt)
  acc_ct <- sum(diag(tabValCt))/sum(tabValCt)
  
  predValsvm <- predict(fitsvm, type = "response", newdata = val_data)
  tabValsvm <- table(val_data$solar_system_count, predValsvm)
  acc_svm <- sum(diag(tabValsvm))/sum(tabValsvm)
  
  
  predValBag <-predict(fitbag,type ="class",newdata =val_data)
  tabValBag <- table(val_data$solar_system_count, predValBag$class)
  acc_bag <- sum(diag(tabValBag))/sum(tabValBag)
  
  # accuracy
  acc <- c(logistic = acc_log, random_forest = acc_rf , class_tree = acc_ct, svm =acc_svm, bagging=acc_bag)
  out[r,1] <- acc_log
  out[r,2] <- acc_rf
  out[r,3] <- acc_ct
  out[r,4] <- acc_svm
  out[r,5] <- acc_bag
  
  
  # use the method that did best on the validation data to predict the test data
  best <- names( which.max(acc) )
  switch(best,
         logistic = {
           pred2 <-predict(fitLog,type ="response",newdata =test_data)# logistic regression
           predTestLog <-ifelse(pred2>0.5,"high","low")
           tabTestLog <- table(test_data$solar_system_count, predTestLog)
           accBest <- sum(diag(tabTestLog))/sum(tabTestLog)
           
           
         },
         random_forest = {
           predTestRf <- predict(fitrf, type = "response", newdata = test_data)
           tabTestRf <- table(test_data$solar_system_count, predTestRf)
           accBest <- sum(diag(tabTestRf))/sum(tabTestRf)
           
          },
         class_tree ={
           predTestCt <- predict(fitCt, type = "class", newdata = test_data)
           tabTestCt <- table(test_data$solar_system_count, predTestCt)
           accBest <- sum(diag(tabTestCt))/sum(tabTestCt)
           
           
         },
         svm ={
           
           predTestsvm <- predict(fitsvm, type = "response", newdata = test_data)
           tabTestsvm <- table(test_data$solar_system_count, predTestsvm)
           accBest <- sum(diag(tabTestsvm))/sum(tabTestsvm)
           },
         bagging={
           predTestbag <- predict(fitbag, type = "class", newdata = test_data)
           tabTestbag <- table(test_data$solar_system_count, predTestbag$class)
           accBest <- sum(diag(tabTestbag))/sum(tabTestbag)
           }
         )
  out[r,6] <- best
  out[r,7] <- accBest
  
  print(r)
}


# check out the error rate summary statistics
table(out[,6])
tapply(out[,7], out[,6], summary)
boxplot(out$test ~ out$best, xlab = "best classifiers", ylab = "test accuracy")
stripchart(out$test ~ out$best, add = TRUE, vertical = TRUE,
           method = "jitter", pch = 19, col = adjustcolor("magenta3", 0.2))



