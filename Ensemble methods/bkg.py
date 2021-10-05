'''
Combine the prediction of several base estimators to improve generalizability/Robustness over a single estimator

Averaging/Bagging methods:  Forests of random trees
    1: Build estimators independently, then average their predictions.
    2: each estimator is a strong and complex (LOW BIAS HIGH VARIANCE) model (e.g. fully developed decision trees).
    3: help REDUCE VARIANCE, but may SLIGHTLY INCREASE BIAS.

    Generate random subsets of a training set for Bagging method:
    Pasting:            draw random subset based on random subset of samples
    Bagging:            draw random subset based on random subset of samples with replacement
    Random Subspace:    draw random subset based on random subset of features
    Random Patches:     draw random subset based on random subset of both features and samples

    RandomForest method:
        1: each tree is a CART tree. input samples (size: m) are collected through bootstrap sampling (size: m)
        2: randomness in the selected features at each decision node. Pick the best feature and splitting feature value
        that corresponds to least Gini impurity

        if classification problem:
            Estimate output class using the plurality vote from T independent CART trees
            For example, the output of sample X is Class A in Tree #1, Class B in Tree #2, and Class B in Tree #3. The
            output will be Class B.
        if regression problem:
            Estimate output would be the sum of outputs from T independent CART trees
            For example, the output of sample X is 4 in Tree #1, 7 in Tree #2, and -2 in Tree #3. The output will be 9.

    Extra-Trees method:
        1: similar to RandomForest method
        2: use all training data (no bootstrap sampling needed) for growing each decision CART tree
        3: randomly choose a feature subset, randomly pick a feature value for each feature, determine best combination
        of feature and feature value by e.g. Gini index OR MSE

        Due to the differences above from RandomForest method
            1: Extra-Trees should generate trees with a greater depth & leaf nodes than RandomForest
            2: Randomly picked feature value leads to greater bias but less variance in training model. Extra-Trees is
            expected to perform better in generalization error

    Totally random trees embedding:
        1: A technique for transforming low-dimensional data to high-dimensional data
        for example: There are 3 CART trees, each tree has 5 leaf nodes. For an input X, you can expect 3 * 5 additional
        features.

    Isolation Forest:
        1: grow decision tree using random feature and feature values for splitting
        2: for a new sample X, determine the probability of an outlier with the averaged depth of affiliated leaf nodes
        among T CART trees


Boosting methods:           AdaBoost, Boosting Tree
    1: Base estimators are built sequentially, aggregate their predictions.
    2: each estimator is a weak and simple (HIGH BIAS LOW VARIANCE) model (e.g. shallow decision trees).
    3: help REDUCE the BIAS of the combined estimators.

    AdaBoost:
        Fit a sequence of weak learner on repeatedly modified versions of the data. Predictions from all estimators are
        combined through a weighted majority vote (or sum) to produce the final prediction
        Apply to both classification and regression problems

        Algorithms:
            for classification:
                SAMME:      Discrete AdaBoost
                SAMME.R:    Real AdaBoost
            for regression:
                R2

    Boosting Tree
        Gradient Boosted Decision Tree (GBDT)
            Generalized boosting to arbitrary differentiable loss functions. Can be used for both classification and
            regression problems.

WARNING:
    unfinished examples:
        Hashing feature transformation using Totally Random Trees
        Manifold learning on handwritten digits: Locally Linear Embedding, Isomap...
        Feature transformations with ensembles of trees
        Gradient Boosting Out-of-Bag estimates
'''

