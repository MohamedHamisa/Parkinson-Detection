import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings  # to show warnings messages
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('parkinsons.data')
df.head()

df.columns
df.describe()
df.info()
df.isnull().sum()
df.shape
df['status'].value_counts()


import seaborn as sns
sns.countplot(df['status'])

df.dtypes


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from xgboost import XGBClassifier  #Extreme Gradient Boosting providing parallel tree boosting in regression and classification 
model = XGBClassifier().fit(X_train, y_train)

predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
accuracy_score(y_test, predictions)

f1_score(y_test, predictions)

#to save model
import pickle
# Writing different model files to file
with open( 'modelForPrediction.sav', 'wb') as f:
    pickle.dump(model,f)
    
with open('standardScalar.sav', 'wb') as f:
    pickle.dump(sc,f)


