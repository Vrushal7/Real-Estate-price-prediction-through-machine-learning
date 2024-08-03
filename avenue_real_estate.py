import pandas as pd
url = 'https://raw.githubusercontent.com/Vrushal7/Real-Estate-price-prediction-through-machine-learning/main/data.csv'
housing = pd.read_csv(url)
print(housing.head())
print(housing.info())
housing['chas'].value_counts()
housing.describe()
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))

#Train test splitting
import numpy as np
np.random.seed(42)
def split_train_test(data,test_ratio):
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(housing,0.2)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")