from sklearn import linear_model, datasets, model_selection, compose, preprocessing, pipeline, metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import pandas as pd

# ex. 1
# print('---------- ex. 1 ----------')
# X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)
# print(X.shape)
#
# N_ALPHAS = 200
# alphas = np.logspace(-10, 2, N_ALPHAS)
# N_FEATURES = X.shape[1]
#
# str_features = np.char.add(np.array(['Feature '] * N_FEATURES), ['%d' % (x + 1) for x in np.arange(10)])
#
# coefs = []
# for a in alphas:
#     ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
#     ridge.fit(X, y)
#     coefs.append(ridge.coef_)
#
# print('size of coefs is {}'.format(np.array(coefs).shape))
#
# ax = plt.gca()
# for i in np.arange(N_FEATURES):
#     ax.plot(alphas, np.array(coefs)[:, i], label=str_features[i])
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])
# print(ax.get_xlim())
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.legend(loc='upper left')
# plt.show()

# ex. 2
print('---------- ex. 2 ----------')
print('Incomplete project due to code errors')
# survey = datasets.fetch_openml(data_id=534, as_frame=True)
# X = survey.data[survey.feature_names]
# pd.set_option('display.max_columns', None)
#
# print(X.describe(include='all'))
# print(X.head())
#
# y = survey.target.values.ravel()
# print(survey.target.head())
# print('dimension of X and y: {0}, and {1}'.format(X.shape, y.shape))
#
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)
# train_dataset = X_train.copy()
# train_dataset.insert(0, 'WAGE', y_train)
# _ = sns.pairplot(train_dataset, kind='reg', diag_kind='kde')
#
# print(survey.data.info())
#
# categorical_columns = ['RACE', 'OCCUPATION', 'SECTOR',
#                        'MARR', 'UNION', 'SEX', 'SOUTH']
# numerical_columns = ['EDUCATION', 'EXPERIENCE', 'AGE']
# preprocessor = compose.make_column_transformer(
#     (preprocessing.OneHotEncoder(drop='if_binary'), categorical_columns), remainder='passthrough'
# )
#
# model = pipeline.make_pipeline(
#     preprocessor,
#     compose.TransformedTargetRegressor(
#         regressor=linear_model.Ridge(alpha=1e-10),
#         func=np.log10,
#         inverse_func=sp.special.exp10
#     )
# )
#
# _ = model.fit(X_train, y_train)
# y_ = model.predict(X_train)
#
# mae = metrics.mean_absolute_error(y_train, y_)
# string_score = f'MAE on training set: {mae:.2f} $/hour'
# y_ = model.predict(X_test)
# mae = metrics.mean_absolute_error(y_test, y_)
# string_score += f'/nMAE on testing set: {mae:.2f} $/hour'
# fig, ax = plt.subplots(figsize=(5, 5))
# plt.scatter(y_test, y_)
# ax.plot([0, 1], [0, 1], 'r--')
# plt.text(3, 20, string_score)
# plt.title('Ridge model, small regularization')
# plt.ylabel('Model Predictions')
# plt.xlabel('Ground Truth')
# plt.xlim([0, 27])
# _ = plt.ylim([0, 27])

# ex. 3
print('---------- ex. 3 ----------')
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_values=True, cv=None, fit_intercept=False)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg)
print(reg.alphas, '\n', reg.alpha_, '\n', reg.cv_values_)
