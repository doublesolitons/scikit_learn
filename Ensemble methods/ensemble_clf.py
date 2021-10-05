import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import datasets, ensemble
from sklearn.tree import DecisionTreeClassifier

n_classes = 3
n_estimators = 30
cmap = plt.cm.RdYlBu
plot_step = .02
plot_step_coarser = .5
RANDOM_SEED = 13

iris = datasets.load_iris()
plot_idx = 1

models = [DecisionTreeClassifier(),
          ensemble.RandomForestClassifier(n_estimators=n_estimators),
          ensemble.ExtraTreesClassifier(n_estimators=n_estimators),
          ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=n_estimators)]

for pair in ([0, 1], [0, 2], [0, 3]):
    for model in models:
        X = iris.data[:, pair]
        y = iris.target

        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        model.fit(X, y)
        scores = model.score(X, y)

        model_title = str(type(model)).split('.')[-1][:-2][:-len('Classifier')]
        print(model_title)

        model_details = model_title

        if hasattr(model, 'estimators_'):
            model_details += ' with {} estimators'.format(len(model.estimators_))
        print(model_details + ' with features', pair, 'has a score of', scores)

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            plt.title(model_title, fontsize=9)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                             np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)

        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors='none')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors.ListedColormap(['r', 'y', 'b']), edgecolor='k', s=20)
        plot_idx += 1

plt.suptitle('Classifiers on feature subsets of the Iris dataset', fontsize=12)
plt.axis('tight')
plt.tight_layout(h_pad=.2, w_pad=.2, pad=2.5)
plt.show()
