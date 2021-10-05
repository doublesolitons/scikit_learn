'''
Linear models means linear combination of the features
y head is the predicted value
'''

from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##########################################################      1.1.1 Ordinary Least Squares (OLS)
# reg = linear_model.LinearRegression()
# x = np.array([[0, 0], [1, 1], [2, 2]])
# y = np.array([0, 1, 2])
# reg.fit(x, y)
# w = reg.coef_
# y_hat = np.dot(x, w)
# print('estimated output {makesure}:{output}'.format(makesure='is', output=y_hat))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(x[:, 0], x[:, 1], y_hat, 'gray')
# ax.scatter(x[:, 0], x[:, 1], y, c='r', marker='o')
# plt.show()

##########################################################      1.1.2 Ridge regression and classification
# reg = linear_model.Ridge(alpha=0.5)
# x = np.array([[0, 0], [0, 0], [1, 1]])
# y = np.array([0, .1, 1])
# reg.fit(x, y)
# print('W0: {W0}\nW1: {W}'.format(W0=reg.intercept_, W=reg.coef_))

############ Ridge coeff = f(regularization)
# X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# print(X)
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# ax.set_title('color map')
# plt.imshow(X)
# ax.set_aspect('equal')
# plt.show()
# y = np.ones(np.shape(X)[0])
# print(y)

# reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
# reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
# print(reg.alpha_)

##########################################################      1.1.3. Lasso
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
print(reg.predict([[1, 1]]))

