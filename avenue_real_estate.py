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

from pandas.plotting import scatter_matrix
attributes=["medv","rm","zn","lstat"]
scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind="scatter",x="rm",y="medv",alpha=0.8)

#Attribute Combinations
housing["taxrm"]=housing["tax"]/housing["rm"]
print(housing.head())
housing.plot(kind="scatter",x="taxrm",y="medv",alpha=0.8)

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
print(imputer.statistics_)
X=imputer.transform(housing)
housing_tr=pd.DataFrame(housing,columns=housing.columns)
housing_tr.describe()

housing=strat_train_set("medv",axis=1)
housing_labels=strat_train_set("medv",axis=1).copy()

#Creating pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])
#numpy array
housing_num_tr=my_pipeline.fit_transform(housing)
housing_num_tr.shape

#Selecting a desired model for Avenue Real Estates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
model=LinearRegression()
model.fit(housing_num_tr,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)

#Evaluate the model
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print(lin_mse)
print(lin_rmse)
