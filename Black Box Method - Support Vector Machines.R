library(kernlab) # For the SVM algorithm

# A Support Vector Machine (SVM) can be imagined as a surface that defines a boundary between various points
# of data which represent examples plotted in multidimensional space according to their feature values.
# The goal of an SVM is to create a flat boundary, called a hyperplane, which leads to fairly homogeneous 
# partitions of data on either side. SVMs combine the strength of the nearest neighbor algorithm as well as
# that of linear regression modelling.

# SVMs can be adapted for use with nearly any type of learning task, including both classification and numeric
# prediction. SVMs are most easily understood when used for binary classification, which is how the method has
# been traditionally applied. In a classifier, the task of the SVM algorithm is to identify a line that
# separates the classes. The algorithm  involves a search for the Maximum Margin Hyperplane (MMH) that creates
# the greatest separation between the two classes. 

# In the case that the data are not linearly separable, the solution to this problem is the use of a slack
# variable, which creates a soft margin that allows some points to fall on the incorrect side of the margin.
# A cost value (denoted as C) is applied to all points that violate the constraints, and rather than finding
# the maximum margin, the algorithm attempts to minimize the total cost.

# Another way of dealing with non-linear separations is the algorithm's ability to map the problem into a
# higher dimension space using a process known as the kernel trick. In doing so, a non-linear relationship may
# suddenly appear to be quite linear. The kernel trick involves a process of adding new features that express
# mathematical relationships between measured characteristics.

# The example below will perform OCR with the SVM algorithm. SVMs are well suited for image data as it can
# learn complex patterns and are not overly sensitive to noise. Moreover, the main weakness of being difficult
# to interpret becomes less relevant in image recognition.

# In this exercise, we'll assume that we have already developed the algorithm to partition the document into
# rectangular regions each consisting of a single character. We will also assume the document contains only
# alphabetic characters in English.

# Since, some of the variables have a wide range, the data needs to be normalized but the SVM algorithm in R
# does the scaling and rescaling automatically. The data has already been randomized during preparation and 
# the first 16000 can be used as training data

letters <- read.csv("datasets/letterdata.csv")
str(letters)

letters_train <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]

#'kernlab' package is used to implement the SVM algorithm. The advantage it has over e1071 or klaR that it is
# written completely in R and is highly customizable. For the first model, a linear kernel is specified, which
# can be done by specifying the value of the kernel as 'vanilladot'

# 83% of the letters are classified correctly

letter_classifier <- ksvm(letter ~ ., data = letters_train, kernel = "vanilladot")
letter_classifier
letter_predictions <- predict(letter_classifier, letters_test)
table(letter_predictions, letters_test$letter)
agreement <- letter_predictions == letters_test$letter
table(agreement)
prop.table(table(agreement))

# In the previous model, a linear kernel was used.  By using a more complex kernel function, we can map the
# data into a higher dimensional space and potentially obtain a better model fit.

# With the Gaussian RBF kernel, we can improve the accuracy to almost 93% which shows the power of the black-
# box algorithm.

letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)
agreement_rbf <- letter_predictions_rbf == letters_test$letter
prop.table(table(agreement_rbf))