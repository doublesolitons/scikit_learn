from sklearn import ensemble, datasets
import matplotlib.pyplot as plt
import collections
import time

now = time.time()
RANDOM_STATE = 123
X, y = datasets.make_classification(n_samples=500, n_features=25,
                                    n_clusters_per_class=1, n_informative=15,
                                    random_state=RANDOM_STATE)
ensemble_clfs = [
    ("RandomForestClassifier, max_features = 'sqrt'",
     ensemble.RandomForestClassifier(warm_start=True, oob_score=True, max_features='sqrt',
                                     random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features = 'log2'",
     ensemble.RandomForestClassifier(warm_start=True, oob_score=True, max_features='log2',
                                     random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features = 'None'",
     ensemble.RandomForestClassifier(warm_start=True, oob_score=True, max_features=None,
                                     random_state=RANDOM_STATE))
]

error_rate = collections.OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 15
max_estimators = 60

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        obb_error = 1 - clf.oob_score_
        error_rate[label].append((i, obb_error))

for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel('n_estimators')
plt.ylabel('OBB error rate')
plt.legend(loc='upper right')
plt.show()
print('it costs %.4f seconds to complete the program' % (time.time() - now))
