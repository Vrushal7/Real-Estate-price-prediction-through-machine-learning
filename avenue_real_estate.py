import pandas as pd
url = 'https://raw.githubusercontent.com/Vrushal7/Real-Estate-price-prediction-through-machine-learning/main/data.csv'
housing = pd.read_csv(url)
print(housing.head())
print(housing.info())
housing['chas'].value_counts()
housing.describe()