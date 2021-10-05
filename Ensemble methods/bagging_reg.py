import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble, tree

n_repeat = 50       # Number of iterations for computer expectations
n_train = 50        # size of training set
n_test = 1000       # size of test set
noise = .1          # std of the noise
np.random.seed(0)

estimators = [('Tree', tree.DecisionTreeRegressor()),
             ('Bagging(Tree)', ensemble.BaggingRegressor(tree.DecisionTreeRegressor()))]
n_estimator = len(estimators)

def f(x):
    """
    ground truth data
    :param x:
    :return:
    """
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise, n_repeat=1):
    """
    generate data with noise
    :param n_samples:
    :param noise:
    :param n_repeat:
    :return:
    """
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))
        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y

X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

plt.figure(figsize=(10, 8))

for n, (name, estimator) in enumerate(estimators):
    y_ = np.zeros((n_test, n_repeat))

    for i in range(n_repeat):       # bagging
        estimator.fit(X_train[i], y_train[i])
        y_[:, i] = estimator.predict(X_test)

    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)
    y_noise = np.var(y_, axis=1)
    y_bias = (f(X_test) - np.mean(y_, axis=1)) ** 2
    y_var = np.var(y_, axis=1)

    print('{0}: {1:.4f}(error) = {2:.4f}(bias^2) + {3:.4f}(var) + {4:.4f}(noise)'
          .format(name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)))

    plt.subplot(2, n_estimator, n + 1)
    plt.plot(X_test, f(X_test), color='b', label='$f(x)$')
    plt.plot(X_train[0], y_train[0], '.b', label='LS ~ $y = f(x) + noise$')

    for i in range(n_repeat):
        if i == 0:
            plt.plot(X_test, y_[:, i], 'r', label=r'$\^y(x)$')
        else:
            plt.plot(X_test, y_[:, i], 'r', alpha=.05)
    plt.plot(X_test, np.mean(y_, axis=1), 'c', label=r'$\mathbb{E}_{LS} \^y(x)$')
    plt.xlim([-5, 5])
    plt.title(name)

    if n == n_estimator - 1:
        plt.legend(loc=(1.1, .5))

    plt.subplot(2, n_estimator, n_estimator + n + 1)
    plt.plot(X_test, y_error, 'r', label='$error(x)$')
    plt.plot(X_test, y_bias, 'b', label='$bias^2(x)$')
    plt.plot(X_test, y_var, 'g', label='$variance(x)$')
    plt.plot(X_test, y_noise, 'c', label='$noise(x)$')

    plt.xlim([-5, 5])
    plt.ylim([0, .1])

    if n == n_estimator - 1:
        plt.legend(loc=(1.1, .5))

plt.subplots_adjust(right=.75)
plt.show()


