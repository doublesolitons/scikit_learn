import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, datasets, utils, ensemble, linear_model, neighbors



# ex. 1
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
print(X.shape)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
print(y.shape)
y[::5, :] += (.5 - rng.rand(20, 2))

regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)
regr_3 = tree.DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

X_test = np.arange(-100.0, 100.0, .01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# plt.figure()
# s = 25
#
# plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
#             edgecolor="black", label="data")
# plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s,
#             edgecolor="black", label="max_depth=2")
# plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s,
#             edgecolor="black", label="max_depth=5")
# plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s,
#             edgecolor="black", label="max_depth=8")
# plt.xlim([-6, 6])
# plt.ylim([-6, 6])
# plt.xlabel("target 1")
# plt.ylabel("target 2")
# plt.title("Multi-output Decision Tree Regression")
# plt.legend(loc="best")
# plt.show()

# ex. 2
data, targets = datasets.fetch_olivetti_faces(return_X_y=True)
train = data[targets < 30]
test = data[targets >= 30]

# limit test set to five samples
n_faces = 5
rng = utils.check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

ESTIMATORS = {
    'Extra trees': ensemble.ExtraTreesRegressor(n_estimators=10, max_features=32,
                                                random_state=0),
    'K-NN': neighbors.KNeighborsRegressor(),
    'Linear regression': linear_model.LinearRegression(),
    'Ridge': linear_model.RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle('Face completion with multi-output estimators', size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title='true faces')
    sub.axis('off')
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray, interpolation='nearest')

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

        sub.axis('off')
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation='nearest')

plt.show()