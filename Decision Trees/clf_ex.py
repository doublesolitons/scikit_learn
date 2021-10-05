from sklearn import tree, datasets
from matplotlib import pyplot as plt

# ex. 1
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
x = [[2, 2]]
# x = [[.4, .4]]
print('Example 1: Predicted output class of {} is {}\n'.format(x, clf.predict(x)))
print('Example 1: Probability to each output class given input {} is {}\n\n'.format(x, clf.predict_proba(x)))


# ex. 2
X, y = datasets.load_iris(return_X_y=True)
iris = datasets.load_iris()
clf = tree.DecisionTreeClassifier().fit(X, y)
tree.plot_tree(clf)

decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X, y)
r = tree.export_text(decision_tree, feature_names=iris['feature_names'])
print('Example 2: Decision Tree text:\n{}'.format(r))
plt.show()

