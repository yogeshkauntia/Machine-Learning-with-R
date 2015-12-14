# Naive Bayes classifier calculates conditional probabilities using Bayes' Theorem. For example,
# in an email classification problem, conditional probabilities such as P(spam|'viagra') can be
# calculated. Multiple conditional probabilities are calculated based on different features which
# are assumed to occur independently. Assumption of equal importance and independence of features
# is one of the biggest drawbacks of the algorithm

# If based on multiple features, the conditional probability of the email being spam is 0.12 and
# it being not spam (ham) is 0.02, a ratio is calculated suggesting that, the likelihood of the
# email being spam is 6 (0.12/0.02) times than it being ham or there's a 85.7% (0.12/(0.12+0.02))
# chance of the email being spam

