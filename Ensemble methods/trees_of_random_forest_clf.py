from sklearn import ensemble, tree, neighbors, model_selection, datasets
import matplotlib.pyplot as plt
import numpy as np
from time import time

# ex. 1
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = ensemble.RandomForestClassifier(n_estimators=10)
# clf = clf.fit(X, y)
# print('---------- Ex. 1 ----------')
# print(clf.predict([[2, 4]]))

# ex. 2
# X, y = datasets.make_blobs(n_samples=1000, n_features=10, centers=100, random_state=0)
#
# clf = tree.DecisionTreeClassifier(random_state=0)
# scores = model_selection.cross_val_score(clf, X, y, cv=5)    # if scoring == None, scoring = estimator's default scorer
# print('---------- Ex. 2 ----------')
# print('Decision Tree classification accuracy: {0: 6.4f}'.format(scores.mean()))
#
# clf = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
# scores = model_selection.cross_val_score(clf, X, y, cv=5)
# print('Random Forest Classification Accuracy: {0: 6.4f}'.format(scores.mean()))
#
# clf = ensemble.ExtraTreesClassifier(n_estimators=10, random_state=0)
# scores = model_selection.cross_val_score(clf, X, y, cv=5)
# print('ExtraTree Classification Accuracy: {0: 6.4f}'.format(scores.mean()))

# ex. 3
# n_jobs = 1
#
# data = datasets.fetch_olivetti_faces()
# X, y = datasets.fetch_olivetti_faces(return_X_y=True)
#
# mask = y < 5
# X, y = X[mask], y[mask]
# print('---------- Ex. 3 ----------')
# print('Fitting ExtraTreesClassifier on faces data with %d cores...' % n_jobs)
# t0 = time()
# forest = ensemble.ExtraTreesClassifier(n_estimators=1000,
#                                        max_features=128,
#                                        n_jobs=n_jobs,
#                                        random_state=0)
# forest.fit(X, y)
# print('done in %.3fs' % (time() - t0))
# importances = forest.feature_importances_
# importances = importances.reshape(int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
#
# plt.matshow(importances, cmap=plt.cm.hot)
# plt.title('Pixel importances with forest of trees')
# plt.show()

# ex. 4
X, y = datasets.make_classification(n_samples=1000,
                                    n_features=10,
                                    n_informative=3,
                                    n_redundant=0,
                                    n_repeated=0,
                                    n_classes=2,
                                    random_state=0,
                                    shuffle=False)
forest = ensemble.ExtraTreesClassifier(n_estimators=20, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print('---------- Ex. 4 ----------\n', 'feature ranking:')
for f in range(X.shape[1]):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title('Feature importances')
plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()