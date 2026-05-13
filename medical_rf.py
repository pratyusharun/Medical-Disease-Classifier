#Import Lib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Dataset
path = "Medical Dataset.csv"
df = pd.read_csv(path)
dataset = pd.DataFrame(df)


#Filter
counts = dataset['diseases'].value_counts()
valid_classes = counts[counts >= 3].index
dataset = dataset[dataset['diseases'].isin(valid_classes)]
dataset = dataset.fillna(0)


#Input & Output
X = dataset.drop('diseases',axis = 1)
y = dataset['diseases']


#Train_Test_Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42, stratify=y)


#Model Training
model = RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


#Metrics
accuracy = accuracy_score(y_test,y_pred)
print(accuracy*100,"%")
print(classification_report(y_test, y_pred))