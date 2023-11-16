# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
1.Read the given Data

2.Clean the Data Set using Data Cleaning Process

3.Apply Feature selection techniques to all the features of the data set

4.Save the data to the file

# CODE
```
Name:PRADEEPASRI S
Reg.No: 212221220038
```
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
sns.heatmap(data.isnull(),cbar=False)
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
data['Embarked']=data['Embarked'].fillna('S')
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.heatmap(data.isnull(),cbar=False)
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')

my_forest.fit(X_train, y_train)

target_predict = my_forest.predict(X_test)
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```
# OUPUT
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/ed4b77ba-6a09-4c42-b94f-091313b01b9c)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/fd992962-a51a-443f-8d78-ca8c66b48156)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/a6066262-2b85-4438-acc2-4df792532f92)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/08b4a59d-1b1e-445e-9735-aac31f66aff6)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/66cd1c92-e384-4c81-87f5-dee7cd6359ba)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/03ded6c0-8e0b-4f9e-8bef-3dc52857f05e)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/933487f2-5c32-4cae-b9c1-7e87df1d6396)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/1dc5d670-9df8-47c4-a376-95b8b4dcffb7)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex-07/assets/131433142/cb5f7ad3-04cf-404f-a9d1-78ac7f9c65dd)
## RESULT
Thus, Successfully performed the various feature selection techniques on a given dataset.







