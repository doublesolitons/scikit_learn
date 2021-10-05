from sklearn import ensemble, datasets, model_selection, tree, metrics
import numpy as np
import matplotlib.pyplot as plt

# ex. 1
print('---------- ex. 1 ----------')

N_ESTIMATORS = 300
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, .1, X.shape[0])

regr_1 = tree.DecisionTreeRegressor(max_depth=4)
regr_2 = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(max_depth=4),
                                    n_estimators=N_ESTIMATORS, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

plt.figure()
plt.scatter(X, y, c='k', label='training samples')
plt.plot(X, y_1, c='g', label='n_estimators=1', linewidth=2)
plt.plot(X, y_2, c='r', label='n_estimators=%d' % N_ESTIMATORS, linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Boosted Decision Tree Regression')
plt.legend(loc='best')
plt.show()

