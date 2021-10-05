import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
print(X.shape)
T = np.linspace(0, 5, 500)[:, np.newaxis]   #
y = np.sin(X).ravel()

y[::5] += 1 * (.5 - np.random.rand(8))

n_neighors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.legend()
    plt.title('KNeighborsRegressors (k = %i, weights = ''%s'')'% (n_neighors, weights))

plt.tight_layout()
plt.show()