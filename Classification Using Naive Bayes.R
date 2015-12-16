library(tm) #For dealing with text data
library(caTools) #For sampling
library(wordcloud) #For wordclouds
library(e1071) #For the Naive Bayes classifier
library(gmodels) #For cross-tabs and result comparison

# Naive Bayes classifier calculates conditional probabilities using Bayes' Theorem. For example,
# in an email classification problem, conditional probabilities such as P(spam|'viagra') can be
# calculated. Multiple conditional probabilities are calculated based on different features which
# are assumed to occur independently. Assumption of equal importance and independence of features
# is one of the biggest drawbacks of the algorithm
# 
# If based on multiple features, the conditional probability of the email being spam is 0.12 and
# it being not spam (ham) is 0.02, a ratio is calculated suggesting that, the likelihood of the
# email being spam is 6 (0.12/0.02) times than it being ham or there's a 85.7% (0.12/(0.12+0.02))
# chance of the email being spam
# 
# A major problem with Naive Bayes is that if a word like 'groceries' has never occured in a spam
# email before, the osterior probability of spam, i.e. P(spam|'groceries') will be zero, and the
# presence of other features will not matter. Solution to this problem is called the Laplace
# estimator. This estimator adds a small number (usually 1) to each of the counts in the frequency
# table to get non-zero probabilities.

# Since we deal with categories and frequency, numeric variables have to be discretized for using
# Naive Bayes classifier. This can be done by binning with appropriate number of bins - not too many,
# and not to less

sms_raw <- read.csv("Datasets/sms_spam.csv", stringsAsFactors = FALSE)
sms_raw$type <- factor(sms_raw$type)
table(sms_raw$type)

sms_corpus <- Corpus(VectorSource(sms_raw$text))
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
corpus_clean <- tm_map(corpus_clean, PlainTextDocument)

sms_dtm <- DocumentTermMatrix(corpus_clean) #Creating a document term matrix
sms_dtm <- removeSparseTerms(sms_dtm, 0.999)
sms_dtm <- as.data.frame(as.matrix(sms_dtm))


set.seed(22)
spl <- sample.split(sms_raw$type, SplitRatio=0.7)

sms_raw_train <- subset(sms_raw, spl==TRUE)
sms_raw_test <- subset(sms_raw, spl==FALSE)

sms_dtm_train <- subset(sms_dtm, spl==TRUE)
sms_dtm_test <- subset(sms_dtm, spl==FALSE)

sms_corpus_train <- subset(sms_corpus, spl==TRUE)
sms_corpus_test <- subset(sms_corpus, spl==FALSE)

wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)
spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

convert_counts <- function(x)
{
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

sms_train <- apply(sms_dtm_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_test, MARGIN = 2, convert_counts)

sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_predictions <- predict(sms_classifier, sms_test)

CrossTable(sms_predictions, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

# The model will be modified using the laplace estimator. This classifies one less ham message as
# spam but 4 more spams as ham
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_predictions2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_predictions2, sms_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
