library(psych) # For multiple histograms in one screen
library(caTools) # For sampling
library(neuralnet) # For implementing ANN

# ANNs were intentionally designed as conceptual models of human brain activity. The model of a single
# artificial neuron can be understood in terms very similar to the biological model.

# Although there are numerous variants of neural networks, each can be defined in terms of the following
# characteristics:
#   1. Activation Function - Transforms neuron's net input signal to single output signal
#   2. Network Topology - Number of neurons, layers and how they are connected
#   3. Training Algorithm - Setting up of connection weights to inhibit or excite neurons based on input

# A major benefit of ANN is allowing information to flow both ways using a recurrent network allowing complex
# patterns to be learned. Adding Delay (short term memory) increases the power of recurrent networks.

# ANN's are the most accurate known approach and make few assumptions about the relationships of the data.
# However, it has a reputation of being computationally intensive and slow to train if we have a complex
# topology. The result is a complex black box model which is difficult or even impossible to interpret.

# Each iteration of the backpropogation algorithm has two phases. A forward phase in which neurons are
# activated in sequence and the output signal is produced (random weights are used). The backward phase uses
# errors (difference between output signals and true value results) and propogates it backwards to readjust
# the weights and reduce further errors.

# The change in weights in each iteration uses the gradient descent algorithm. This algorithm changes the
# weights and reduces the error at the "learning rate" of the algorithm. If the changes are very small, it will
# take the algorithm a long time to find the minima and if it is large, the change might overshoot the minima

# The example models the strength of concrete using artificial neural networks.

concrete <- read.csv("datasets/concrete.csv")
str(concrete)

# ANN requires standardization and the built-in R function scale() can be used if the underlying data is
# normal. Otherwise, uniform standardization is advisable.
multi.hist(concrete)
normalize <- function(x)
{
  return((x - min(x)) / (max(x) - min(x)))
}
concrete_norm <- as.data.frame(lapply(concrete, normalize))
summary(concrete_norm$strength)

spl <- sample.split(concrete_norm$strength, SplitRatio=0.75)
concrete_train <- subset(concrete_norm, spl==TRUE)
concrete_test <- subset(concrete_norm, spl==FALSE)

# Because it ships as part of the standard R installation, the nnet package is perhaps the most frequently
# cited ANN implementation. It uses a slightly more sophisticated algorithm than standard backpropagation.
# Another strong option is the RSNNS package, which offers a complete suite of neural network functionality,
# with the downside being that it is more difficult to learn. We will use the package 'neuralnet'

# The first model is a simple multilayer feedforward network with only a single hidden node. The results are
# good and there's a 0.81 correlation between predicted and actual values. Mean absolute error is also low.
# A neural network with just one hidden node is similar to linear regression - the weights of the input nodes
# are equivalent to the beta coefficients and the weight of the hidden node is equivalent to the intercept

formula <- strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age

concrete_model <- neuralnet(formula, data = concrete_train)
plot(concrete_model)

model_results <- compute(concrete_model, concrete_test[1:8])
predicted_strength <- model_results$net.result

cor(predicted_strength, concrete_test$strength)
MAE <- function(actual, predicted)
{
  mean(abs(actual - predicted))
}
MAE(concrete_test$strength, predicted_strength)

# As networks with more complex topologies are capable of learning more difficult concepts, let's see what
# happens when we increase the number of hidden nodes to five. We use the neuralnet() function as before, but
# add the parameter hidden = 5

# The model sure looks more complex and the SSE (error as seen on the plot) has decreased considerably and
# the number of steps have increased. The correlation increased from 0.81 to 0.92 and the MAE has decreased
# from 0.09 to only 0.05

concrete_model2 <- neuralnet(formula, data = concrete_train, hidden = 5)
plot(concrete_model2)

model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result

cor(predicted_strength2, concrete_test$strength)
MAE(concrete_test$strength, predicted_strength2)