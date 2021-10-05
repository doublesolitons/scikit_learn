import time
import pandas as pd
# from tqdm import tqdm, tnrange

# numbers = list(range(0, 3))
#
# for current_number in tqdm(numbers, desc='Looping over an array'):
#     time.sleep(2)
#     print('\n', current_number)

def update_df(df: pd.DataFrame,
              clf: str,
              acc: float,
              remarks: list[str] = [],
              split: float = .5):#
    new_row = {'Classifier': clf,
               'Accuracy': acc,
               'split_size': split,
               'Remarks': remarks}
    df = df.append(new_row, ignore_index=True)
    return df

df = pd.DataFrame(columns= ['Classifier', 'Accuracy', 'Split_size', 'Remarks'])
df = update_df(df, 'KNN', 76, .1)
df = update_df(df, 'SVM', 99, remarks=['Check again'])
df = update_df(df, 'LR', .7, remarks=['param tuning', 'overfitting'])
print(df)
