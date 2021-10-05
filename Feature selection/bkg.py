from sklearn import feature_selection, datasets, svm, linear_model
import numpy as np
import matplotlib.pyplot as plt
import time

# ex. 1
print('---------- ex. 1 ----------')
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))

# ex. 2 Univariate feature selection, select features based on univariate statis. tests
print('---------- ex. 2 ----------')
X, y = datasets.load_iris(return_X_y=True)
print('dimension of iris data {}'.format(X.shape))
# print(X[:5, :])
X_new = feature_selection.SelectKBest(feature_selection.chi2, k=2).fit_transform(X, y)
print('dimension of filtered iris data {}'.format(X_new.shape))

# ex. 3 Recursive feature elimination,
# print('---------- ex. 3 ----------')
# digits = datasets.load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# y = digits.target
#
# svc = svm.SVC(kernel='linear', C=1)
# rfe = feature_selection.RFE(estimator=svc, n_features_to_select=10, step=1)#
# rfe.fit(X, y)
# print(rfe.ranking_.shape)
# print(np.size(np.unique(rfe.ranking_)))
# ranking = rfe.ranking_.reshape(digits.images[0].shape)
# print(digits.images[0].shape)
# print(digits.target[0])
#
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title('Ranking of pixels with RFE')
# plt.show()

# ex. 4 SelectFromModel
print('---------- ex. 4 ----------')
print('L1-based')
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
print(diabetes.DESCR)

lasso = linear_model.LassoCV().fit(X, y)
importance = np.abs(lasso.coef_)

feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title('Feature importance via coefficients')
# plt.show()

threshold = np.sort(importance)[-3] + .01
tic = time.time()
sfm = feature_selection.SelectFromModel(lasso, threshold=threshold).fit(X, y)
toc = time.time()

print('Feature selected by SelectFromModel: '
      f'{feature_names[sfm.get_support()]}')
print(f'Done in {toc - tic:.3f}s')

print('Tree-based')


plt.show()
