"""
Supervised Non-parametric method, can be used for classification and regression

ID3 algorithm:
    Find each node in a greedy manner
    Apply to categorical feature
    metrics: determine the order of feature selection based on information gain, but ig favors the features with more
    values, due to larger entropy.
C4.5 algorithm:
    Apply to both categorical and continuous feature values
    metrics: normalized information gain
    support missing attribute values. Missing values are not used in gain and entropy calculations
    handel attributes with different costs
C5.0 algorithm:
    faster, less memory usage, smaller decision tree
    support boosting and weighting different cases and misclassification types
CART algorithm:
    metrics:        Gini impurity (similar to entropy) for classification problem, improve calculation efficiency
                    Mean square (or Absolute) error for regression problem
    node-leaves:    Always conduct binary classification at each node. No matter continuous or discrete values, each
                    attribute has more than 1 opportunity to make classification decision
    Pruning:        1: generate pruned trees from trained tree
                    2: Select pruned tree using cross validation
Adv:
    simple to understand and interpret (white box model)
    little data preparation. No need for normalization, need dummy variables, no blank values
    O(log(n))
    can handle numerical and categorical data
    can handle multi-output problems
Dis-Adv:
    Overfitting. Need pruning
    Unstable model due to small variation in training data
    Trained model is often towards local optimal decision, but not global optimization
    Biased estimator if some classes dominate
Complexity:
    cost at each node to find the feature with largest information gain: O(n_features)
    cost to construct a balanced binary tree: O(m * n * log(m))
    Total cost over entire tree: O(m ^^ 2 * n * log(m))
Tips:
    if n > m, the estimator is likely to overfit

    Use dimensionality reduction (PCA, ICA, Feature selection) to find features that are discriminative

    Visualize tree by using export function, use max_depth = 3 as initial tree depth

    Required number of samples doubles for each additional level of a tree

    Control leaf size before and after a split can help overfitting

    Balance data is needed. one way is to sampling equal number of samples from each class, the other is to normalizing
    the sum of the sample weights for each class to the same value

    If the input matrix X is very sparse, best to convert it to space csc_matrix before calling fit and csr_matrix
    before classing predict.
"""