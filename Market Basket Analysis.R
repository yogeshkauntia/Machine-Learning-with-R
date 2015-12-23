library(arules) #The association rules package is used for market basket anlaysis

# The machine learning methods for identifying associations among items in transactional data is called
# 'Market Bucket Analysis'.
# Developed in the context of Big Data and database science, association rules are not used for prediction, but
# rather for unsupervised knowledge discovery in large databases.

# Because association rule learners are unsupervised, there is no need for the algorithm to be trained; data
# does not need to be labeled ahead of time. The program is simply unleashed on a dataset in the hope that
# interesting associations are found. The downside, of course, is that there isn't an easy way to objectively
# measure the performance of a rule learner.

# The most-widely used approach for efficiently searching large databases for rules is known as Apriori.
# Whether or not an association rule is deemed interesting is determined by two statistical measures:
# support and confidence. By providing minimum thresholds for each of these metrics and applying the Apriori
# principle, it is easy to drastically limit the number of rules reported, perhaps even to the point where
# only the obvious, or common sense, rules are identified.

# Market basket analysis is used behind the scenes for the recommendation systems used in many brick-and-mortar
# and online retailers.

# read.transactions is used instead of read.csv which results in the creation of a sparse matrix

groceries <- read.transactions("datasets/groceries.csv", sep = ",")
summary(groceries)
inspect(groceries[1:5])
itemFrequency(groceries[, 1:3])
itemFrequencyPlot(groceries, support = 0.1) # Items with at least 10% support
itemFrequencyPlot(groceries, topN = 20) # Top 20 frequent items
image(sample(groceries, 100)) # For visualizing the sparse matrix

# The default support is 0.1 which implies that the rule must be valid in 983 cases, therefore 0 rules. When
# support and confidence parameters are changed, 423 rules surface. The basket size ranges from 2 to 4. Lift
# is the ratio of having an association over just being in the basket by chance
apriori(groceries)
groceryrules <- apriori(groceries, parameter = list(support = 0.006, confidence = 0.25, minlen = 2))
summary(groceryrules)
inspect(groceryrules[1:3])

# There are usually three types of rules: trivial, inexplicable and actionable. A trivial rule may be diapers-
# -formula. An inexplicable rule is something which the data suggests but maybe just noise and should not be
# paid attention to. Actionable rules are the hidden gems, which when discovered may make sense but was not
# obvious otherwise. Sorting by lift gives some interesting rules

inspect(sort(groceryrules, by = "lift")[1:5])

# Association rules for just one item can be visualized using the subset function.
# Additional operators are available for partial matching (%pin%) and complete matching (%ain%). Partial 
# matching allows you to find both citrus fruit and tropical fruit using one search: items %pin% "fruit".
# Complete matching requires that all listed items are present. The association rules can be written to csv
# or converted to a dataframe.

berryrules <- subset(groceryrules, items %in% "berries")
inspect(berryrules)

groceryrules_df <- as(groceryrules, "data.frame")
