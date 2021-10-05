import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = [['Steve', 25, 55], ['Linus', 28, 60], ['Elon', 26]]
df = pd.DataFrame(data=data, columns=['Name', 'Age', 'Weight'])
print('Initial data entries:\n{0}\n'.format(df))
print('Check the results of N/A values:\n{0}\n'.format(df.isnull()))
print('total number of N/A values:\n{0}\n'.format(df.isnull().sum()))
print('Data entries after removing N/A values:\n{0}'.format(df.dropna()))

# method 1: use SimpleImputer module
imputer = SimpleImputer(missing_values=np.nan)
imputer = imputer.fit(df[['Weight']])
df['Weight'] = imputer.transform(df[['Weight']])

# method 2: use fillna function
df = df.fillna(df.mean())
print(df)

# use standard scaler for normalizing features
# following case transformed data into the # of std from the mean of sample
std = StandardScaler()
X = std.fit_transform(df[['Age', 'Weight']])
print('# of std shifting away from sample mean:\n{}'.format(X))

# now deal with categorical variables
df_cat = pd.DataFrame([['Green', 'M', 10.1, 'Class 1'],
                       ['Blue', 'L', 20.1, 'Class 2'],
                       ['White', 'M', 30.1, 'Class 1']])
df_cat.columns = ['color', 'size', 'price', 'class_label']
print('\n{}\n'.format(df_cat))

# method 1
# size_mapping = {'M': 1, 'L': 2}
# df_cat['size'] = df_cat['size'].map(size_mapping)

# method 2
# class_le = LabelEncoder()
# df_cat['class_label'] = class_le.fit_transform(df_cat['class_label'].values)

# avoid multicollinearity
df_cat = pd.get_dummies(df_cat[['color', 'size', 'price']], drop_first=True)
print('transform size label to number:\n{}\n'.format(df_cat))
print(df_cat.columns)


