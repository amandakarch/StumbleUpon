library(readr)
source('C:/Users/akarch1/Desktop/Spring20/QTM3/Data/BabsonAnalytics.R')
df = read_csv('C:/Users/akarch1/Desktop/Spring20/QTM3/Data/stumbleupon.csv')

# Managing & cleaning data
summary(df)
df = df[-c(1674),] # gets rid of weird value for logical target -> not 0 or 1 (and had ? as a value in another column)
df$label = as.factor(df$label) # Target value

# We narrowed it down to two models: kNN and classification trees

# Run this code only for kNN (gets rid of categorical inputs):
df$url = NULL
df$urlid = NULL
df$boilerplate = NULL
df$alchemy_category = NULL
df$alchemy_category_score = NULL
df$framebased = NULL
df$hasDomainLink = NULL
df$is_news = NULL
df$lengthyLinkDomain = NULL
df$news_front_page = NULL

# Run this code for both kNN and classification trees:
df$url = as.factor(df$url)
df$urlid = as.factor(df$urlid)
df$boilerplate = as.factor(df$boilerplate)
df$alchemy_category = as.factor(df$alchemy_category)
df$alchemy_category_score = as.factor(df$alchemy_category_score)
df$framebased = as.factor(df$framebased)
df$hasDomainLink = as.factor(df$hasDomainLink)
df$is_news = as.factor(df$is_news)
df$lengthyLinkDomain = as.factor(df$lengthyLinkDomain)
df$news_front_page = as.factor(df$news_front_page)

# Partitioning data
N = nrow(df)
trainingSize = round(0.6*N) 
trainingCases = sample(N, trainingSize)
training = df[trainingCases, ]
test = df[-trainingCases, ]

# kNN model
library(class)
k_best = kNNCrossVal(label ~ ., training) # controlling for overfitting with k_best
predictions_kNN = kNN(label ~ ., training, test, k_best)
table(predictions_kNN, test$label)
error_rate_kNN = sum(predictions_kNN!=test$label)/nrow(test)

# Classification tree model
library(rpart)
library(rpart.plot)
tree = rpart(label ~ ., data = training)
rpart.plot(tree)
predictions_tree = predict(tree, test, type = "class")
error_rate_tree = sum(predictions_tree!=test$label)/nrow(test)
stoppingRulesOverfit = rpart.control(minsplit = 1, minbucket = 1, cp = 0)
overfit = rpart(label~., data = training, control = stoppingRulesOverfit)
pruned = easyPrune(overfit) # controlling for overfitting
predictions_pruned = predict(pruned, test, type = "class")
error_rate_pruned = sum(predictions_pruned != test$label)/nrow(test)

# Benchmark error rate - compare all error rates against this
error_bench = benchmarkErrorRate(training$label, test$label)
