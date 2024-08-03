from joblib import dump, load
import  numpy as np
model=load('AvenueRealEstates.joblib')

model.predict()