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
def split_train_test(data,test_ratio):
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)