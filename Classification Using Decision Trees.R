library(caTools) #For sampling
library(gmodels) #For cross-tabs and results
library("C50") #For running the C5.0 decision tree algorithm

# Decision trees are an all-purpose classifier and uses the most important features for classification.
# Its drawbacks are using only feature at a time or being prone to over-fitting.
# 
# The most important challenge faced by a decision tree model is which feature to split on. Features which
# provide splits with examples primarily of a single class are more desirable. If all the values are from
# the same class - it is said to be pure.
# One such measure of purity is 'entropy' used by the C5.0 decision tree algorithm. The value of entropy
# ranges from 0 (pure) to 1 (mixed). The algorithm measures the change in entropy of each split calling
# the metric 'information gain'.

# Pruning a decision tree is important because a tree can grow to an extent where only pure splits exist.
# Such a tree is over-fitted to the training data and a tree needs to be pruned so that it generalizes
# better to unseen data. One way to do it is stopping the tree from growing when it reaches certain number
# of decisions or when all nodes have small number of examples. This is called "pre-pruning". The other
# way is to grow a large tree and then reduce the tree size using error rates at the nodes as the pruning
# criteria. This technique, "post-pruning" is more effective.

# The C5.0 algorithm post-prunes by default and can do it by either cutting off a branch or moving the
# branch further up. This process of grafting branches is called "subtree raising". The C5.0 algorithm
# is used to identify risky bank loans in this example.


credit <- read.csv("Datasets/credit.csv")
str(credit)
credit$default <- as.factor(credit$default)
table(credit$default)


set.seed(22)
spl <- sample.split(credit$default, SplitRatio=0.9)
credit_train <- subset(credit, spl==TRUE)
credit_test <- subset(credit, spl==FALSE)

# The model is only 73% accurate on test-data which is marginally higher than 70% accuracy which can be
# obtained by simply tagging all entries as 'Not-default'. The model performs particularly worse in case
# of 'default' cases. Out of 30 actual defaults, only 12 are categorized as defaults which can be very
# expensive to the lender.

credit_model <- C5.0(credit_train[-21], credit_train$default)
summary(credit_model)
credit_pred <- predict(credit_model, credit_test)
CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# To improve the performance of the model, boosting methods can be used and one such method for C5.0 is
# changing the number of trials

# Boosting increased the accuracy slightly - from 73% to 80% but the performance for the 'default' cases
# is still bad. In this situation, the cost of not identifying a potential defaulter is much higher than
# not giving a loan to a border line non-defaulter. The C5.0 algorithm allows us to assign cost to the
# classification errors.

credit_boost10 <- C5.0(credit_train[-21], credit_train$default, trials = 10)
credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# If we determine that a false negative is four times more expensive than a false positive, an error cost
# matrix can be built as shown in the code below. The correct assignments (diagonals) will obviously have
# 0 cost

# The overall accuracy of the model decreases when compared to the boosted model. The overall accuracy is
# 67% which is lower than if we just assign all as non-defaulters. However, 93% of the defaulters are now
# correctly identified

error_cost <- matrix(c(0, 1, 4, 0), nrow = 2)
credit_cost <- C5.0(credit_train[-21], credit_train$default, costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))