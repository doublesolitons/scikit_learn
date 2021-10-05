from sklearn import ensemble, datasets, model_selection, tree, metrics
import numpy as np
import matplotlib.pyplot as plt


# ex. 1
# print('---------- ex. 1 ----------')
# X, y = datasets.load_iris(return_X_y=True)
# clf = ensemble.AdaBoostClassifier(n_estimators=100)
# score = model_selection.cross_val_score(clf, X, y, cv=5)
# print('---------- ex. 1 ----------')
# print('Averaged performance of AdaBoost model is %3.2f' % score.mean())

# ex. 2
# print('---------- ex. 2 ----------')
# N_ESTIMATORS = 400
# LEARNING_RATE = 1
# X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
# X_train, y_train = X[:2000], y[:2000]
# X_test, y_test = X[2000:], y[2000:]
#
# dt_stump = tree.DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)         # depth is up to 1 layer
# dt_stump.fit(X_train, y_train)
# dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)
#
# dt = tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)               # depth is up to 9 layers
# dt.fit(X_train, y_train)
# dt_err = 1.0 - dt.score(X_test, y_test)
#
# ada_discrete = ensemble.AdaBoostClassifier(base_estimator=dt_stump,
#                                            learning_rate=LEARNING_RATE,
#                                            n_estimators=N_ESTIMATORS,
#                                            algorithm='SAMME')
# ada_discrete.fit(X_train, y_train)
#
# ada_real = ensemble.AdaBoostClassifier(base_estimator=dt_stump,
#                                        learning_rate=LEARNING_RATE,
#                                        n_estimators=N_ESTIMATORS,
#                                        algorithm='SAMME.R')
# ada_real.fit(X_train, y_train)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.plot([1, N_ESTIMATORS], [dt_stump_err] * 2, 'k-', label='Decision Stump Error')
# ax.plot([1, N_ESTIMATORS], [dt_err] * 2, 'k--', label='Decision Tree Error')
#
# ada_discrete_error = np.zeros((N_ESTIMATORS,))
# for i, y_ in enumerate(ada_discrete.staged_predict(X_test)):
#     ada_discrete_error[i] = metrics.zero_one_loss(y_, y_test)
#
# ada_discrete_error_train = np.zeros((N_ESTIMATORS,))
# for i, y_ in enumerate(ada_discrete.staged_predict(X_train)):
#     ada_discrete_error_train[i] = metrics.zero_one_loss(y_, y_train)
#
# ada_real_error = np.zeros((N_ESTIMATORS,))
# for i, y_ in enumerate(ada_real.staged_predict(X_test)):
#     ada_real_error[i] = metrics.zero_one_loss(y_, y_test)
#
# ada_real_error_train = np.zeros((N_ESTIMATORS,))
# for i, y_ in enumerate(ada_real.staged_predict(X_train)):
#     ada_real_error_train[i] = metrics.zero_one_loss(y_, y_train)
#
# ax.plot(np.arange(N_ESTIMATORS) + 1, ada_discrete_error, 'r', label='Discrete AdaBoost Test Error')
# ax.plot(np.arange(N_ESTIMATORS) + 1, ada_discrete_error_train, 'b', label='Discrete AdaBoost Train Error')
# ax.plot(np.arange(N_ESTIMATORS) + 1, ada_real_error, 'r--', label='Real AdaBoost Test Error')
# ax.plot(np.arange(N_ESTIMATORS) + 1, ada_real_error_train, 'b--', label='Real AdaBoost Train Error')
#
# ax.set_ylim((0.0, 0.5))
# ax.set_xlabel('n_estimators')
# ax.set_ylabel('error rate')
#
# leg = ax.legend(loc='upper right', fancybox=True)
# leg.get_frame().set_alpha(.7)
#
# plt.show()


# ex. 3
# print('---------- ex. 3 ----------')
# X, y = datasets.make_gaussian_quantiles(n_samples=13000, n_features=10, n_classes=3, random_state=1)
#
# N_SPLIT = 3000
#
# X_train, X_test = X[:N_SPLIT], X[N_SPLIT:]
# y_train, y_test = y[:N_SPLIT], y[N_SPLIT:]
#
# bdt_real = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),
#                                        n_estimators=600, learning_rate=1)       # default algorithm: SAMME.R
# bdt_distrete = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),
#                                            n_estimators=600, learning_rate=1.5, algorithm='SAMME')
# bdt_real.fit(X_train, y_train)
# bdt_distrete.fit(X_train, y_train)
#
# real_test_errors = []
# discrete_test_errors = []
#
# for real_test_predict, discrete_test_predict in zip(bdt_real.staged_predict(X_test), bdt_distrete.staged_predict(X_test)):
#     real_test_errors.append(1 - metrics.accuracy_score(y_test, real_test_predict))
#     discrete_test_errors.append(1 - metrics.accuracy_score(y_test, discrete_test_predict))
#
# n_trees_discrete = len(bdt_distrete)
# n_trees_real = len(bdt_real)
#
# discrete_estimate_errors = bdt_distrete.estimator_errors_[:n_trees_discrete]
# real_estimate_errors = bdt_real.estimator_errors_[:n_trees_real]
# discrete_estimate_weights = bdt_distrete.estimator_weights_[:n_trees_discrete]
# real_estimate_weights = bdt_real.estimator_weights_[:n_trees_real]
#
# plt.figure(figsize=(15, 5))
# plt.subplot(131)
# plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, 'b', label='SAMME', alpha=.5)
# plt.plot(range(1, n_trees_real + 1), real_test_errors, 'r', label='SAMME.R', alpha=.5)
# plt.legend()
# plt.ylim([.18, .62])
# plt.ylabel('Test Error')
# plt.xlabel('Number of Trees')
#
# plt.subplot(132)
# plt.plot(range(1, n_trees_discrete + 1), discrete_estimate_errors,
#          "b", label='SAMME', alpha=.5)
# plt.plot(range(1, n_trees_real + 1), real_estimate_errors,
#          "r", label='SAMME.R', alpha=.5)
# plt.legend()
# plt.ylabel('Error')
# plt.xlabel('Number of Trees')
# plt.ylim((.2,
#          max(real_estimate_errors.max(),
#              discrete_estimate_errors.max()) * 1.2))
# plt.xlim((-20, len(bdt_distrete) + 20))
#
# plt.subplot(133)
# plt.plot(range(1, n_trees_discrete + 1), discrete_estimate_weights,
#          "b", label='SAMME')
# plt.plot(range(1, n_trees_real + 1), real_estimate_weights,
#          "r", label='SAMME.R')
# plt.legend()
# plt.ylabel('Weight')
# plt.xlabel('Number of Trees')
# plt.ylim((0, discrete_estimate_weights.max() * 1.2))
# plt.xlim((-20, n_trees_discrete + 20))
#
# # prevent overlapping y-axis labels
# plt.subplots_adjust(wspace=0.25)
# plt.show()


# ex. 4
print('---------- ex. 4 ----------')
X1, y1 = datasets.make_gaussian_quantiles(cov=2.,
                                          n_samples=200, n_classes=2, random_state=1)
X2, y2 = datasets.make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                          n_samples=300, n_classes=2, random_state=1)

X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

bdt = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1),
                                  algorithm='SAMME', n_estimators=200)
bdt.fit(X, y)

plot_colors = 'br'
plot_step = .02
class_name = 'AB'

plt.figure(figsize=(10, 5))

plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

for i, n, c in zip(range(2), class_name, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolors='k', label='Class %s' %n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Decision Boundary')

twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())

plt.subplot(122)
for i, n, c in zip(range(2), class_name, plot_colors):
    plt.hist(twoclass_output[y == i], bins=10, range=plot_range,
             facecolor=c, label='Class %s' % n, alpha=.5, edgecolor='k')

x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Score')

plt.tight_layout()
plt.subplots_adjust(wspace=.35)
plt.show()




