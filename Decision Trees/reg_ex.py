from sklearn import tree

# ex. 1
X = [[0, 0], [2, 2]]
y = [.5, 2.5]
clf = tree.DecisionTreeRegressor().fit(X, y)
x= [[1., 1.]]
print('Example 1: Predicted output of {} is {}\n'.format(x, clf.predict(x)))

# ex. 2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
print(np.shape(X), np.shape(y))
y[::5] += 3 * (0.5 - rng.rand(16))

regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

X_test = np.arange(0.0, 5., .01)[:, np.newaxis]
print(np.shape(X_test))
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y, s=20, edgecolors='black',
            c='darkorange', label='data')
plt.plot(X_test, y_1, color='black',
         label='max_depth=2', linewidth=2, alpha=.5)
plt.plot(X_test, y_2, color='red',
         label='max_depth=5', linewidth=2, alpha=.5)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()


