from sklearn import ensemble, metrics, datasets, model_selection, inspection
import numpy as np
import matplotlib.pyplot as plt

# ex. 1
# print('---------- ex. 1 ----------')
# X, y = datasets.make_friedman1(n_samples=1200, random_state=0, noise=1.0)
# X_train, X_test = X[:200], X[200:]
# y_train, y_test = y[:200], y[200:]
#
# est = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=.1,
#                                          max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
# print('%.3f' % metrics.mean_squared_error(y_test, est.predict(X_test)))

# ex. 2
print('---------- ex. 2 ----------')

N_ESTIMATORS = 500
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.1, random_state=13)
reg = ensemble.GradientBoostingRegressor(learning_rate=.01, min_samples_split=5,
                                         max_depth=4, n_estimators=N_ESTIMATORS)
reg.fit(X_train, y_train)

mse = metrics.mean_squared_error(y_test, reg.predict(X_test))
print('The mean squared error on test set: {:.4f}'.format(mse))

test_score = np.zeros((N_ESTIMATORS,), dtype=np.float64)
for i, y_ in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_)

fig = plt.figure(figsize=(6, 6))
plt.subplot(111)
plt.title('Deviance')
plt.plot(np.arange(N_ESTIMATORS) + 1, reg.train_score_, 'b-',
         label='Training set deviance')
plt.plot(np.arange(N_ESTIMATORS) + 1, test_score, 'r-',
         label='Test set deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
# plt.show()

# feature importance
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.barh(pos, feature_importance[sorted_idx])
plt.yticks(pos, np.array(datasets.load_diabetes().feature_names)[sorted_idx])
plt.title('Feature importance (MDI)')

result = inspection.permutation_importance(reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(122)
plt.boxplot(result.importances[sorted_idx].T, vert=False,
            labels=np.array(datasets.load_diabetes().feature_names)[sorted_idx])
plt.title('Permutation Importance (test set)')
fig.tight_layout()
plt.show()

