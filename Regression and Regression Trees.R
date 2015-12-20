# Linear Regression -------------------------------------------------------
library(psych) # Informative scatterplot matrices

# The example predicts medical expenses using linear regression. 'Charges' in the insurance dataset is
# equivalent to medical expenses and the dependent variable for the analysis.

insurance <- read.csv("datasets/insurance.csv")
str(insurance)
summary(insurance$charges)
hist(insurance$charges)

# Correlation with numeric variables to check linear relationships
cor(insurance[c("age", "bmi", "children", "charges")])

# Scatterplot matrix using the 'psych' package. This provides scatterplots, correlations values and ellipses
# The curve on the plot is 'loess smooth'
pairs.panels(insurance[c("age", "bmi", "children", "charges")])

ins_model <- lm(charges ~ ., data = insurance)
summary(ins_model)

#Model diagnostics are performed by examining residual plots and testing for homoscedasticity
plot(ins_model)

# Residuals fail the normality test - visually and through Shapiro-Wilk test
qqPlot(ins_model, main="QQ Plot")
shapiro.test(ins_model$residual)

# Homoscedasticity tests - ncvTest performs the Breusch-Pagan test and the null hypothesis is that the
# variance is constant over the values of the response (fitted values). The test fails with high confidence
# showing that the variances are not constant
ncvTest(ins_model)

# Multicollinearity is also tested using vif commands
vif(ins_model)
sqrt(vif(ins_model)) > 2 # If true, there's a problem

# Non-linearity. The main problem seems to occur with age and bmi. Age seems to have a concave structure and
# a higher order transformation may help. BMI doesn't seem to have a linear relationship but clusters of BMI
# have different residual variances suggesting that the variable should be categorized
crPlots(ins_model)
plot(ins_model$residual,insurance$age)
plot(ins_model$residual,insurance$bmi)


# Diagnostics of the model is ignored and following the book, some sample transformation of independent
# variables are carried out. For any analysis, transformation comes after carrying diagnostic tests and
# examining residual plots.

# Higher order transformation
insurance$age2 <- insurance$age^2

# Converting to binary indicator or binning of numeric variables. If we have a hunch that the effect of a
# feature is not cumulative but an effect is seen when a specific threshold is reached, binnging can be
# performed.

# For instance, BMI may have zero impact on medical expenditures for individuals in the normal weight
# range, but it may be strongly related to higher costs for the obese (that is, BMI of 30 or above)
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# If certain features are believed to have a combined impact on the dependent variable, it is included in
# the model with a '*' sign between the variables. R automatically adds the variables individually

# All the three transformed variable are significant as seen in the model summary below. The interactive
# impact of high bmi and smoking is very high as expected. The new model has non-normal residuals but the
# heteroscedasticity has been addressed. Remedial measures to address non-normality of residuals need to be
# performed before the model can be considered correct and useful.

ins_model2 <- lm(charges ~ age + age2 + children + bmi + sex +
                   bmi30*smoker + region, data = insurance)
summary(ins_model2)
plot(ins_model2)
shapiro.test(ins_model2$residual)
ncvTest(ins_model2)


# Regression Trees --------------------------------------------------------
library(rpart) #For regression trees
library(rpart.plot) #For visualizing the trees
library(caTools) #For sampling
library(RWeka) #For building a model tree using the M5-prime algorithm

# Despite the name, regression trees do not use linear regression for prediction. The average value of the
# examples at the leaf nodes are used. Second type of trees for numeric prediction is called Model Trees.
# These are grown like regression trees but at each leaf, a multiple linear regression model is built from
# the examples reaching that node. Depending on the number of leaf node, tens or hundreds of regression
# models may be built. Regression trees are grown like classification trees and each split tries to increase
# the homogeneity of the nodes ('standard deviation reduction' is a common splitting criterion)

# Model trees have numerous advantages over linear regression as it combines the strengths of decision
# trees and linear regression. Automatic feature selection allows usage of large number of features.
# However, it is more difficult to interpret and it is difficult to have one statistic to measure the
# performance of the model.

wine <- read.csv("datasets/whitewines.csv")
str(wine)
spl <- sample.split(wine$quality, SplitRatio=0.75)
wine_train <- subset(wine, spl==TRUE)
wine_test <- subset(wine, spl==FALSE)

m.rpart <- rpart(quality ~ ., data = wine_train)
summary(m.rpart)
rpart.plot(m.rpart, digits = 3)
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)
p.rpart <- predict(m.rpart, wine_test)

# Performance can be measured by looking at summary statistics or more accurately by mean absolute error

# Summary statistics suggest that the predictions are not good at extremes but the bulk of the predictions
# should be close to the actual values
summary(p.rpart)
summary(wine_test$quality)

# Mean of absolute error suggests that on an average the prediction deviates by only 0.588 points from the
# actual value
MAE <- function(actual, predicted)
{
  mean(abs(actual - predicted))
}

MAE(wine_test$quality, p.rpart)

# A 'model tree' is built to improve the performance. M5-prime algorithm in the RWeka package can be used to
# build a model tree with the M5P() function. The performance increased marginally with mean absolute error
# reducing from 0.588 to 0.543

m.m5p <- M5P(quality ~ ., data = wine_train)
summary(m.m5p)
p.m5p <- predict(m.m5p, wine_test)
MAE(wine_test$quality, p.m5p)
