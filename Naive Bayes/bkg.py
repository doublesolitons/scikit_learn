"""
Joint probability: P(X, y). The probability that X and y are taking a specific pair of values
Conditional probability: P(X|y). The probably of taking a specific X value given fixed y value
P(y|X) is the probability of y taking class value Ck given X value
P(y|X) * P(X) = P(X|y) * P(y)
    P(y|X) = P(X|y) * P(y) / P(X), where P(X) is a constant here

if all features are approximately independent of each other, P(X|y) = P(X_1|y) * P(X_2|y) * ... * P(X_n|y)

Estimated class of y
    y_ = argmax(y) P(y) * P(X_1|y) * P(X_2|y) * ... * P(X_n|y)

The major question for Naive Bayes method is the assumptions of distribution for each feature.
If X_i is continuous,
    P(X_i|y) follows Gaussian distribution
if X_i is discrete,
    P(X_i|y) follows multinomial distribution, the relative frequency counting for feature i given class Ck
    = N_yi / N_y, where N_yi is the number of times feature i appears in a sample of class y in a training set
                        N_y is the total count of appearance of all features for class y in a training set
    Specifically,
        Complement Naive Bayes considers the imbalance of data between classes
        Bernoulli Native Bayes specifically penalizes the features with zero frequency counting. Situations include:
            a) binary outcome for every single feature values,
            b) large # of feature values compared with training sample size
"""

from sklearn import datasets, model_selection, naive_bayes
# ex. 1
print('---------- ex. 1 ----------')

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.5, random_state=0)
gnb = naive_bayes.GaussianNB()
y_ = gnb.fit(X_train, y_train).predict(X_test)
print('Number of mislabeled points out of a total %d points: %d'
      % (X_test.shape[0], (y_ != y_test).sum()))
