library(dplyr) #For data-manipulation
library(class) #For using the kNN classifier
library(gmodels) #For cross-tabs and result comparison

# K-Nearest Neighbor (kNN) is a lazy learning algorithm. It is 'lazy learning' because
# generalization or abstraction does not occur. Since instances from the training set are
# use, this algorithm has two major disadvantages:
#   1. Doesn't perform well with a noisy training set
#   2. The entire training set needs to be stored


# To classify a new test point, 'k' nearest training points are found based on Eucledian distance
# and a vote is done. The test point belongs to the category with the maximum votes.
# Choosing k is a balance between overfitting and underfitting the data (bias-variance tradeoff)


# The example below deals with classification of lumps on breasts tissues as malignant
# or benign. This will be done using kNN algorithm on measurements of these biopsied cells.

wbcd <- read.csv("Datasets/wisc_bc_data.csv", stringsAsFactors = FALSE)

wbcd <- select(wbcd, -id)
table(wbcd$diagnosis)
wbcd$diagnosis <- as.factor(wbcd$diagnosis)

summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# kNN algorithm heavily depends on calculating the Euclidean distance between the new point
# and instances of the training set. Therefore, it is required that the variables used for the
# distances are in the same scale. Therefore, normalization is used

normalize <- function(x)
{
  return ((x - min(x)) / (max(x) - min(x)))
}

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))

wbcd_train <- wbcd_n[1:469, ] #Training set
wbcd_test <- wbcd_n[470:569, ] #Test set

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

# k = 21 because it is closed to the square root of number of observations in the training set
# and an odd number will not result in tie of votes

wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k=21)

# The classifier works great. However, the 2 errors where malignant tumors are classified as
# benign may prove very expensive to the patient who may believe the tumor to be non-cancerous

CrossTable(x = wbcd_test_labels, y = wbcd_test_pred)

# An alternative is using the z-transformation
wbcd_z <-  as.data.frame(scale(wbcd[-1]))

# And then the same steps..

wbcd_train <- wbcd_z[1:469, ] #Training set
wbcd_test <- wbcd_z[470:569, ] #Test set

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k=21)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred)

# Different values of 'k' may be tried to get the desired level of false positives and negatives.
# However, it is important to know that over-fitting has to be avoided