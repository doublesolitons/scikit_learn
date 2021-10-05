from sklearn import datasets, ensemble
import numpy as np
import matplotlib.pyplot as plt


# ex. 1
# print('---------- ex. 1 ----------')
# X, y = datasets.make_hastie_10_2(random_state=0)
# X_train, X_test = X[:2000], X[2000:]
# y_train, y_test = y[:2000], y[2000:]
#
# clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.,
#                                           max_depth=1, random_state=0).fit(X_train, y_train)
# print(clf.score(X_test, y_test))


# ex. 2
print('---------- ex. 2 ----------')
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
X = X.astype(np.float32)

labels, y = np.unique(y, return_inverse=True)
# print('labels shape: {0} %n y shape: {1}'.format(labels.shape, y.shape))
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

original_params = {
    'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2, 'min_samples_split': 5
}

plt.figure()

for label, color, setting in [
    ('No shrinkage', 'orange', {'learning_rate': 1.0, 'subsample': 1.0}),
    ('learning_rate=0.1', 'turquoise', {'learning_rate': 0.1, 'subsample': 1.0}),
    ('subsample=0.5', 'blue', {'learning_rate': 1.0, 'subsample': 0.5}),
    ('learning_rate=0.1, subsample=0.5', 'gray', {'learning_rate': 0.1, 'subsample': 0.5}),
    ('learning_rate=0.1, max_features=2', 'magenta', {'learning_rate': 0.1, 'max_features': 2})]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_ in enumerate(clf.staged_decision_function(X_test)):
        test_deviance[i] = clf.loss_(y_test, y_)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', color=color, label=label)

plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')
plt.show()
