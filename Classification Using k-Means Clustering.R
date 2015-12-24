library(stats) # kmeans() within the package can be used to implement the k-means algorithm

# Clustering is an unsupervised machine learning task that automatically divides the data into clusters, or
# groupings of similar items. It does this without having been told what the groups should look like ahead
# of time. Clustering is guided by the principle that records inside a cluster should be very similar to
# each other, but very different from those outside.

# If you begin with unlabeled data, you can use clustering to create class labels. From there, you could apply
# a supervised learner such as decision trees to find the most important predictors of these classes. This is
# called "semi-supervised learning".

# k-means require an initial guess as to how many clusters naturally occur in the data. This can be confirmed
# once the algorithms are run with different values of k. Choosing the number of clusters requires a delicate
# balance. Setting the k to be very large will improve the homogeneity of the clusters, and at the same time,
# it risks overfitting the data. Square root of (n/2) is one rule of thumb or the correct number of clusters
# can be found by finding the elbow point in a graph of within-group homogeneity vs k. Since the learning is
# unsupervised, getting the right 'k' may not be very important. It is common to see that more homogeneous
# groups stick together while less homogeneous groups form and disband as 'k' is altered.

teens <- read.csv("datasets/snsdata.csv")
str(teens)
table(teens$gender, useNA = "ifany")

# Since the data is for high school goers, ages like 3 or 106 do not make sense and must be pruned

summary(teens$age)
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, teens$age, NA)

# Solving the missing data problem with dummy coding. Dummy coding involves creating a separate binary 1 or 0
# valued dummy variable for each level of a nominal feature except one

teens$female <- ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0)

# Solving the missing data problem with imputation. Using the grad year as a dummy for age, the mean age of
# each graduation year can be used

aggregate(data = teens, age ~ gradyear, mean, na.rm = TRUE)
ave_age <- ave(teens$age, teens$gradyear, FUN = function(x) mean(x, na.rm = TRUE))
teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
summary(teens$age)

# Clusters will be initially built just on interests of the teenagers which range from variable 5-40. Just
# like the kNN algorithm, they are normalized so that any feature doesn't dominate the distance based
# algorithm by its scale
interests <- teens[5:40]
interests_z <- as.data.frame(lapply(interests, scale))

# In the 1985 John Hughes coming-of-age comedy, The Breakfast Club, high-school-age characters are identified
# in terms of five stereotypes: a Brain, an Athlete, a Basket Case, a Princess, and a Criminal.
# Given that these identities prevail throughout popular teen fiction, five seems like a reasonable starting
# point for k.

teen_clusters <- kmeans(interests_z, 5)
teen_clusters$size
teen_clusters$centers
teens$cluster <- teen_clusters$cluster

# Confirming clusters formed using interests using demographics
aggregate(data = teens, age ~ cluster, mean)
aggregate(data = teens, female ~ cluster, mean)
aggregate(data = teens, friends ~ cluster, mean)
