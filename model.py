import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection   import train_test_split

df = pd.read_csv("train.csv")
df.drop(columns=["Id"],inplace=True)

df.drop(columns=df.columns[df.isnull().sum().values>200],inplace=True)
df.dropna(inplace=True)
df.isnull().sum().values

replace = df["MSZoning"].dtype

for column in df.columns:
    if df[column].dtype == replace:
        uniques = np.unique(df[column].values)
        for idx,item in enumerate(uniques):
            df[column] = df[column].replace(item,idx)
            
df = (df - df.mean()) / df.std()
df["bias"] = np.ones(df.shape[0])

trainingData, testingData = train_test_split(df, test_size=0.3) 
print ("Training data:", trainingData.count(), "; Testing data:", testingData.count())
training_y = trainingData["SalePrice"].values
training_X = trainingData.drop(columns=["SalePrice"]).values
testing_y = testingData["SalePrice"].values
testing_X = testingData.drop(columns=["SalePrice"]).values

# Create and fit the model
lr = LogisticRegression(fit_intercept=False)
lr.fit(training_X.astype('int'),training_y.astype('int'))
print(lr.coef_)
# Calculate Mean Absolute Error (Easier to interpret than MSE)
mean_absolute_error(training_y.astype('int'), training_X.dot(lr.coef_))
# Predict
lr_predict = lr.predict(testing_X.astype('int'))
# Accuracy
accuracy = accuracy_score(testing_y.astype('int') ,lr_predict)
print("Model Accuracy")
print(accuracy)

