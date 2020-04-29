library(readr)
source('C:/Users/akarch1/Desktop/Spring20/QTM3/Data/BabsonAnalytics.R')
df = read_csv('C:/Users/akarch1/Desktop/Spring20/QTM3/Data/stumbleupon.csv')

### MANAGING AND CLEANING DATA ###

# changing variable types
df$label = as.factor(df$label) # Target value

df$url = as.factor(df$url)
df$boilerplate = as.factor(df$boilerplate)

df$alchemy_category = as.factor(df$alchemy_category)
df$hasDomainLink = as.factor(df$hasDomainLink)
df$is_news = as.factor(df$is_news)
df$lengthyLinkDomain = as.factor(df$lengthyLinkDomain)
df$news_front_page = as.factor(df$news_front_page)

df$alchemy_category_score = as.numeric(df$alchemy_category_score)

# replacing ?s with NAs & fixing NAs
#df$alchemy_category = gsub("?", NA, df$alchemy_category, fixed=TRUE)
df$is_news = gsub("?",NA,df$is_news, fixed = TRUE)
df$news_front_page = gsub("?", NA, df$news_front_page, fixed=TRUE)

summary(df)

# Columns with NAs or ?s:
# alchemy_category
levels(df$alchemy_category)
library(ggplot2)
ggplot(df, aes(alchemy_category))+geom_bar()
#df$alchemy_category[is.na(df$alchemy_category)] = 'unknown'

# alchemy_category_score
library(naniar)
df$alchemy_category_score = impute_mean(df$alchemy_category_score)

# is_news
df$is_news[is.na(df$is_news)] = 0

# news_front_page
df$news_front_page[is.na(df$news_front_page)] = 0

# label - row 1674
df$label[1674] = impute_mean(df$label[1674])

# Other Notes:
# avglinksize has a max of 363 - check that out
# compression_ratio has a max of 21 - check that out
# image_ratio has a max of 113 and a negative number for min - check that out
# library(caret)
# normalizer6 = preProcess(df[,6], method='range')
# normalizer11 = preProcess(df[,11], method='range')
# normalizer17 = preProcess(df[,17], method='range')
# df = predict(normalizer6, df)
# df = predict(normalizer11, df)
# df = predict(normalizer17, df)

# figure out how to slice out base URL name for url - could be a better way to predict
library(tidyverse)
library(tidytext)
library(data.table)
df$url = df$url %>%
  str_replace_all('http://','') %>%
  str_replace_all('/.*','')

# Deleting columns
df$urlid = NULL # tells us nothing - just a classifier
df$framebased = NULL # every value is the same

# changing categorical to factors - ensuring nothing changed amongst them during cleaning

df$url = as.factor(df$url)
df$boilerplate = as.factor(df$boilerplate)

df$alchemy_category = as.factor(df$alchemy_category)
df$hasDomainLink = as.factor(df$hasDomainLink)
df$is_news = as.factor(df$is_news)
df$lengthyLinkDomain = as.factor(df$lengthyLinkDomain)
df$news_front_page = as.factor(df$news_front_page)

### PARTITION DATA ###
set.seed(1) # ensures reproducibility
N = nrow(df)
trainingSize = round(0.6*N) 
trainingCases = sample(N, trainingSize)
training = df[trainingCases, ]
test = df[-trainingCases, ]

### kNN ###
library(dplyr)
df_knn = select_if(df, is.numeric)
df_knn$label = df$label
training_knn = df_knn[trainingCases, ]
test_knn = df_knn[-trainingCases, ]
library(class)
k_best = kNNCrossVal(label ~., training_knn)
predictions_kNN = kNN(label ~., training_knn, test_knn, k_best)
table(predictions_kNN, test$label)
error_rate_kNN = sum(predictions_kNN!=test$label)/nrow(test)

### Classification tree ###
library(rpart)
tree = rpart(label ~ ., data = training)
predictions_tree = predict(tree, test, type = "class")
error_rate_tree = sum(predictions_tree!=test$label)/nrow(test)
stoppingRulesOverfit = rpart.control(minsplit = 1, minbucket = 1, cp = 0)
overfit = rpart(label~., data = training, control = stoppingRulesOverfit)
pruned = easyPrune(overfit)
predictions_pruned = predict(pruned, test, type = "class")
error_rate_pruned = sum(predictions_pruned != test$label)/nrow(test)

### New model: naive Bayes' ###
library(e1071)
nb = naiveBayes(label~url+boilerplate+alchemy_category+hasDomainLink+is_news+lengthyLinkDomain+news_front_page, data=training)
predictions = predict(nb, test)
table(predictions, test$label)
error_rate_nb = sum(predictions!=test$label)/nrow(test)

### Bagging ###
library(randomForest)
rf = randomForest(label~.-url-boilerplate, data=training, ntree=500)
predictions_rf = predict(rf, test)
error_rf = sum(predictions_rf != test$label)/nrow(test)
importance(rf)

### Stacking ###
predictions_kNN_full = kNN(label~., training_knn, df_knn, k_best)
predictions_tree_full = predict(tree, df)
predictions_rf_full = predict(rf, df)
df_stacked = cbind(df, predictions_rf_full, predictions_tree_full, predictions_kNN_full )
training_stacked = df_stacked[trainingCases,]
test_stacked = df_stacked[-trainingCases,]
stacked = rpart(label~., data=training_stacked)
predictions_stacked = predict(stacked, test_stacked, type='class')
error_stacked = sum(predictions_stacked != test_stacked$label)/nrow(test_stacked)

# benchmark error rate:
error_bench = benchmarkErrorRate(training$label, test$label)
