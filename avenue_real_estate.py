import pandas as pd
url = 'https://raw.githubusercontent.com/Vrushal7/Real-Estate-price-prediction-through-machine-learning/main/data.csv'
housing = pd.read_csv(url)
print(housing.head())
print(housing.info())
housing['chas'].value_counts()
housing.describe()

#For plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))

#Train test splitting
#for learning purpose
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

#train_set,test_set=split_train_test(housing,0.2)
#print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")

from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['chas']):
    strat_train_set= housing.loc[train_index]
    strat_test_set= housing.loc[test_index]

strat_test_set['chas'].describe()

#Looking for correlations
corr_matrix=housing.corr()
corr_matrix['medv'].sort_values(ascending=False)

#from pandas.plotting import scatter_matrix