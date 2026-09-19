import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Day":['saturday','sunday','monday','tusday','wensday','thursday','friday'],
    "Timegame":[2,3,6,4,1,2,3],
    "Timestudy":[3,2,5,1,9,1,1]
}


df = pd.DataFrame(data)

print(df)

x = np.arange(len(df['Day']))
width = 0.35

plt.bar(x - width/2,df['Timegame'],width,label = "game",color='skyblue')
plt.bar(x + width/2,df['Timestudy'],width,label='study',color='lightgreen')


plt.bar(df['Day'],df['Timegame'],df['Timestudy'],color='skyblue')
plt.title('what do you do in free time')
plt.xlabel("Day")
plt.ylabel("Hours")
plt.xticks(x,df['Day'])
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = {
    "StudyHours": [2,4,6,8,10,12],
    "GameHours": [8,6,4,2,1,0],
    "Passexam":[0,0,0,1,1,1],
}

df = pd.DataFrame(data)

x = df[["StudyHours","GameHours"]]
y = df["Passexam"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "StudyHours": [2,4,6,8,10,12],
    "Passexam": [0,0,0,1,1,1],
}

df = pd.DataFrame(data)

x = df[["StudyHours"]]
y = df["Passexam"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

for i in range(len(df)):
    if df["Passexam"][i] ==1:
        plt.scatter(df["StudyHours"][i],df["Passexam"][i],color = "green",label = "Pass" if i ==0 else "")
    else:
        plt.scatter(df["StudyHours"][i],df["Passexam"][i],color="red",label = "Fail" if i ==0 else "")

plt.title("Pass in exam")
plt.xlabel("StudyHours")
plt.ylabel("Pass in exam")
plt.legend()
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

x = df[iris.feature_names]
y = df['species']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)


model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

plt.scatter(df['sepal length (cm)'],df['sepal width (cm)'],c=df['species'],cmap='viridis')
plt.xlabel('species Lengh (cm)')
plt.ylabel('species Width (cm)')
plt.title('Iris Dataset Visualization')
plt.show()

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['species'] = iris.target

print(df.head())

x = df[iris.feature_names]
y = df['species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

plt.scatter(df["sepal length (cm)"],df["sepal width (cm)"],c=df['species'],cmap='viridis')
plt.xlabel('species length (cm)')
plt.ylabel('species width (cm)')
plt.title("Iris Dataset")
plt.show()

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['species'] = iris.target

x = df[iris.feature_names]
y = df['species']


x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('Accurcay:',accuracy_score(y_test,y_pred))

plt.scatter(df['petal length (cm)'],df['petal width (cm)'],c=df['species'],cmap='viridis')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('iris dataset - knn visualization')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())  

le = LabelEncoder()
df['Sex'] = le.fit_transform(df["Sex"])

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df['Fare'].fillna(df["Fare"].mean())

x = df[["Age", "Sex", "Pclass", "Fare"]]   
y = df["Survived"]  


x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=10
)
for k in [1,3,5,7,9,11]:
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Accuracy:",accuracy_score(y_test,y_pred))


import pandas as pd
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt

rl = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(rl)

print(df.head())

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

x = df[["Age","Sex","Fare","Pclass"]]
y = df["Survived"]

k_values = [3,5,7,9,11]
accuracies = []

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

for k in [3,5,7,9,11]:
    model = KNeighborsClassifier(n_neighbors=k)
    models = SVC()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker= 'o')
for i ,acc in enumerate(accuracies):
    plt.text(k_values[i],accuracies[i],f"{acc:.2f} ",ha='center',va='bottom')
plt.xlabel('k (number of nighbors)')
plt.ylabel('Accuracy')
plt.title("Titanic knn accuracy vs k")
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

rl = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(rl)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

x = df[["Age","Sex","Fare","Pclass"]]
y = df["Survived"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


print("Accuracy:",accuracy_score(y_test,y_pred))
print("Class report:",classification_report(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))

modelk = KNeighborsClassifier(n_neighbors=7)
modelk.fit(x_train,y_train)
y_pred = modelk.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Class report:",classification_report(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

rl = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(rl)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

x = df[["Age","Fare","Sex","Pclass"]]
y = df["Survived"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=11)

model = RandomForestClassifier(max_depth=7,random_state=11,n_estimators=500,max_features='log2')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Random Forest model:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("class report:",classification_report(y_test,y_pred))

model = DecisionTreeClassifier(max_depth=5,random_state=11,min_samples_split=4,criterion='gini')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Decision model:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("class report:",classification_report(y_test,y_pred))


model = KNeighborsClassifier(n_neighbors=13,metric='manhattan',weights='uniform')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("KN model:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("class report:",classification_report(y_test,y_pred))

param_grid = {
    'n_estimators':[500,1000],
    'max_depth':[3,5,7,9,10,13,15,17,19],
    'max_features':['sqrt','log2']
}
grid = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
grid.fit(x_train,y_train)
print("Best param:",grid.best_params_)
print("Best score:",grid.best_score_)

param_grid = {
    'max_depth':[3,5,7,9,11,13,15,17,19],
    'min_samples_split':[2,4,6],
    'criterion':['gini','entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
grid.fit(x_train,y_train)
print("Best params:",grid.best_params_)
print("best score:",grid.best_score_)

grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)
grid.fit(x_train,y_train)
print("Best params:",grid.best_params_)
print("Best score:",grid.best_score_)


param_grid ={
    'C':[10,1,0.1,0.01],
    'penalty':['l1','l2','elastincnet'],
    'solver':['saga','liblinear','lbfgs']
}
grid = GridSearchCV(LogisticRegression(max_iter=1000),param_grid,cv=5)
grid.fit(x_train,y_train)
print("Logistic model:")
print('Best params:',grid.best_params_)
print('Best score:',grid.best_score_)

param_grid = {
    'n_neighbors': [3,5,7,9,11,13],
    'metric':['euclidean','manhattan'],
    'weights':['uniform','distance']
}
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("class report:",classification_report(y_test,y_pred))


from sklearn.ensemble import GradientBoostingClassifier,RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ssl
import  urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
ulr = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(ulr,encoding='latin1')

print(df.head())
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())


x = df[["Age","Sex","Fare","Pclass"]]
y = df["Survived"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

model = RandomForestClassifier()


model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Gradient boosting:")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Class report:",classification_report(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))

XGB = RandomForestRegressor()
XGB.fit(x_train,y_train)
y_predict = XGB.predict(x_test)
print("XGB:")
print("Accuracy:",accuracy_score(y_test,y_predict))
print("Class report:",classification_report(y_test,y_predict))
print("Matrix:",confusion_matrix(y_test,y_predict))
log_clf = LogisticRegression(max_iter = 1000)
tree_clf = ExtraTreeClassifier(max_depth=5,random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=13,metric='manhattan',weights = 'uniform')
rf_clf = RandomForestClassifier(max_depth=7,n_estimators=500,max_features='log2',random_state=42)

voting_clf = VotingClassifier(
    estimators = [('lr', log_clf), ('dt', tree_clf), ( 'knn',knn_clf),('rf',rf_clf),],
    voting='hard'
)
voting_clf.fit(x_train,y_train)
y_pred = voting_clf.predict(x_test)

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression

ulr = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(ulr)

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())


# df["AgeGroup"] = LabelEncoder().fit_transform(df["AgeGroup"].astype(str))

df["Familysize"] = df["SibSp"] + df["Parch"] + 1
df["AgeGroup"] = pd.cut(df["Age"], bins = [0,12,18,40,60,80], labels = [0,1,2,3,4])
df['IsAlone'] = (df['Familysize'] == 1).astype(int)
df["Farebin"] = pd.qcut(df['Fare'],4,labels =  [0,1,2,3])

x = df[['Age','Fare','Sex','Pclass','Familysize','AgeGroup',"IsAlone","Farebin"]]
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)


rf = RandomForestClassifier(max_depth=7,n_estimators=500,max_features='log2',random_state=42)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)


print("Accuray:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("Class report:",classification_report(y_test,y_pred))

import matplotlib.pyplot as plt

importance= rf.feature_importances_
featurs = x.columns


plt.barh(featurs,importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feture')
plt.title('Random forest feature importance')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder


import requests

from io import StringIO

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url, verify=False)  # SSL check off
df = pd.read_csv(StringIO(response.text))


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Age"] = df["Age"].fillna(df["Age"].mean())

x = df[['Age','Fare','Pclass','Sex']]
y = df['Survived']

model = RandomForestClassifier(max_depth=7,n_estimators=500,random_state=42,max_features='log2')

score = cross_val_score(model,x,y,cv=5,scoring='accuracy')

print('Cross valution scores:',score)
print('Mean accuracy:',score.mean())


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


url = "titanic.csv"
df = pd.read_csv(url)


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

X = df[["Age","Sex","Pclass","Fare"]]
y = df["Survived"]


rf = RandomForestClassifier(max_depth=7, n_estimators=500, max_features='log2', random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("Random Forest:")
print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

xg = XGBClassifier()
score = cross_val_score(xg,X,y,cv=5,scoring='accuracy')
print("XGBoots:")
print("Cross validation score:", score)
print("Mean Accuracy:", score.mean())

lg = LogisticRegression(max_iter=1000,random_state=42)
score= cross_val_score(lg,X,y ,cv=5, scoring="accuracy")
print("Logistic Regression:")
print("Cross Validation Score:",score)
print("Mean Score:",score.mean())

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score




params = {
    'hours:' [1,2,3,4],
    'study:' [10,20,30,40]
}

df  = pd.DataFrame(params)

x = df[['hours']]
y = df['study']


x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

rt = RandomForestClassifier()
rt.fit(x_train,y_train)
y_pred = rt.predict(x_test)

print(accuracy_score(y_test,y_pred))

new_hours = 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

url = 'titanic.csv'
df = pd.read_csv(url)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df["Fare"] = df['Fare'].fillna(df["Fare"].mean())

x = df[["Age","Fare",'Pclass','Sex']]
y = df['Survived']

rf = RandomForestClassifier(max_depth=7,n_estimators=500,max_features='log2',random_state=42)

train_size,train_score,test_score = learning_curve(rf,x,y,cv=5,scoring='accuracy',
                                                   train_sizes=np.linspace(0.1,1.0,10))
train_mean = np.mean(train_score,axis = 1)
test_mean = np.mean(test_score,axis=1)

plt.plot(train_size,train_mean,label = "Training Accuracy")
plt.plot(train_size,test_mean,label = "Testing Accuracy")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.title("learning Curve - random forest")
plt.legend()
plt.show()

xg = XGBClassifier()
train_size,train_score,test_score = learning_curve(xg,x,y,cv=5,scoring='accuracy',
                                                   train_sizes=np.linspace(0.1,1.0,10))

train_mean = np.mean(train_score,axis = 1)
test_mean = np.mean(test_score,axis = 1)


plt.plot(train_size,train_mean,label = "Training Accuracy")
plt.plot(train_size,test_mean,label = 'Test Accuracy')
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.title("Learning Curve - XGBoost")
plt.legend()
plt.show()

lg = LogisticRegression()

train_size,train_score,test_score = learning_curve(lg,x,y,cv=5,scoring='accuracy',
                                                   train_sizes=np.linspace(0.1,1.0,10))

train_mean = np.mean(train_score,axis =1)
test_mean = np.mean(test_score , axis = 1)

plt.plot(train_size,train_mean,label = "Training Accuracy")
plt.plot(train_size,test_mean,label = "Test Accuracy")
plt.xlabel("Training size")
plt.ylabel("Accuracy")
plt.title("Learning curve - Logistic")
plt.legend()
plt.show()

from sklearn.metrics import classification_report,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder


url = "titanic.csv"
df = pd.read_csv(url)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

x = df[["Age","Sex",'Fare',"Pclass"]]
y  = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

models = {
    "RandomForest": RandomForestClassifier(max_depth=7,n_estimators=500,max_features='log2',random_state=42),
    "XGBoost" : XGBClassifier(use_label_encoder = False,eval_metric = 'logloss'),
    "Logistic Regression": LogisticRegression(max_iter = 1000)
}

for name, model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]

    print(f"/n{name}")
    print(classification_report(y_test,y_pred))
    print("ROC-ACU",roc_auc_score(y_test,y_pred))


from sklearn.metrics import auc,roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt


url = 'titanic.csv'
df = pd.read_csv(url)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df["Age"] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df["Fare"].fillna(df["Fare"].mean())

x = df[['Age','Fare','Pclass','Sex']]
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


models = {
    'RandomForest': RandomForestClassifier(max_depth=7,random_state=42,max_features='log2',n_estimators=500),
    'XGBoost': XGBClassifier(),
    'Logistic': LogisticRegression(max_iter=1000)
}



plt.figure(figsize=(12,5))
for name, model in models.items():
    model.fit(x_train,y_train)
    y_proba = model.predict_proba(x_test)[:,1]
    fpr,tpr,_ = roc_curve(y_test,y_proba)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label = f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False positive Rate')
plt.ylabel('True positive Rate')
plt.title('Roc curve comparison')
plt.legend()
plt.show()

plt.figure(figsize = (12,5))
for name,model in models.items():
    model.fit(x_train,y_train)
    y_proba = model.predict_proba(x_test)[:,1]
    precision,recall,_ = precision_recall_curve(y_test,y_proba)
    plt.plot(recall,precision,label = name)

plt.xlabel('Recall')
plt.ylabel('precision')
plt.title('precision-recall curve comparison')
plt.legend()
plt.show()    

from sklearn.metrics import auc,roc_curve,precision_recall_curve,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'titanic.csv'
df = pd.read_csv(url)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df["Age"] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df["Fare"].fillna(df["Fare"].mean())

x = df[['Age','Fare','Pclass','Sex']]
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(max_depth=7,random_state=42,max_features='log2',n_estimators=500)
model.fit(x_train,y_train)

y_prob = model.predict_proba(x_test)[:,1]

theshold = np.arange(0.1,0.9,0.1)
for t in theshold:
    y_pred_thresh = (y_prob >= t).astype(int)
    precision = precision_score(y_test,y_pred_thresh)
    recall = recall_score(y_test,y_pred_thresh)
    f1 = f1_score(y_test,y_pred_thresh)
    print(f"Threshold {t:.1f} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = 'titanic.csv'
df = pd.read_csv(url)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

x = df[['Age','Fare','Pclass','Sex']]
y = df["Survived"]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

rf = RandomForestClassifier(max_depth=7,n_estimators=500,random_state=42,max_features='log2')
rf.fit(x_train,y_train)
y_prob = rf.predict_proba(x_test)[:,1]

threshold = [0.3,0.5,0.7]
for t in threshold:
    y_pred_thresh = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_test,y_pred_thresh)

    plt.figure(figsize=(4,3))
    sns.heatmap(cm,annot=True,fmt="d",cmap='Blues',xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual'])  
    plt.title(f"Confusion Matrix (Threshold={t})")
    plt.show()  

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sbs
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(max_depth=7,max_features='log2',random_state=42,n_estimators=500)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred,target_names = iris.target_names))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(4,3))
sbs.heatmap(cm,annot = True,cmap= 'Blues',xticklabels = iris.target_names,yticklabels = iris.target_names)
plt.title("confusion_matrix - multi-class")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

x , y = make_classification(n_samples=1000, n_features=10,n_classes = 3,
    n_informative=5,weights=[0.7,0.2,0.1],random_state = 42)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=42)
smote = SMOTE(random_state=42)
x_resampled,y_resampled = smote.fit_resample(x_train,y_train)

model = RandomForestClassifier(class_weight='balanced')
model.fit(x_resampled,y_resampled)
y_pred = model.predict(x_test)

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot= True,fmt = "d",cmap= "Blues")
plt.title("Confusion Matrix - Imbalanc Mulite-class")
plt.xlabel("Predicted")
plt.ylabel("Acutal")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

precision_before = [0.88, 0.87, 0.83]
recall_before = [0.98,0.78,0.44]
f1_before = [0.93,0.82,0.58]

precision_after = [0.91,0.83,0.75]
recall_after = [0.94,0.81,0.62]
f1_after = [0.93,0.82,0.68]

classes = ['class1','class2','class3']
x = np.arange(len(classes))
width = 0.35
plt.figure(figsize = (4,3))
plt.bar(x - width / 2, precision_before, width, label = "Before Smote")
plt.bar(x + width / 2, precision_after, width, label = 'After Smote')
plt.xticks(x,classes)
plt.ylabel('precision')
plt.title('Precision comparison Before vs after smote')
plt.legend()
plt.show()

plt.figure(figsize=(4,3))
plt.bar(x - width / 2,f1_before,width,  label = 'before smote')
plt.bar(x + width / 2,f1_before,width,  label = 'after smote')
plt.xticks(x, classes)
plt.ylabel('f1_score')
plt.title('f1 comparison before vs after smote')
plt.legend()
plt.show()



plt.figure(figesize=(4,3))
plt.bar(x - width / 2,recall_before,width, label = 'before smote')
plt.bar(x + width / 2,recall_after,width,label = 'after smote')
plt.xticks(x, classes)
plt.ylabel('recall_score')
plt.title('recall comparison before vs after smote')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
f1_before = [0.93,0.82,0.58]


f1_after = [0.93,0.82,0.68]

classes = ['class 1','calsses 2','classes 3']

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize= (8,5))
plt.bar(x - width / 2,f1_before,width, label = "Before Smote")
plt.bar(x + width / 2,f1_after, width, label = "After Smote")
plt.xticks(x, classes)
plt.ylabel("F1 Score")
plt.title("F1 comparison Before vs After Smote")
plt.legend()
plt.show()


from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import Label_binarize
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

x ,y= make_classification(n_samples = 1000, n_features = 20, n_classes = 3,
n_informative = 10, weights = [0.6,0.3,0.1])

y_bin = Label_binarize(y, classes = [0,2,2])
n_classes = y_bin.shape [1]

x_train,x_test, y_train.y_test = train_test_split(x,y_bin,test_size = 0.3,random_state = 42)

model = RandomForestClassifier()
model.fit(x_train,hy_train)
y_score = model.predict_proba(x_test)


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_calsses):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i],y_score[i][:,1])
    roc_auc[i] -= acu(fpr[i],tpr[i])

plt.figure(figsize = (8,6))
for i in range(n_classes):
    plt.plot(fpr[i], trp[i], label = f'class {i} (ACU = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False positive Rate")
plt.ylable("True positive Rate")
plt.title("ROc Curve - multi-class")
plt.legend()
plt.show()          

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# ساخت دیتاست چندکلاسه
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                           n_informative=10, weights=[0.6, 0.3, 0.1],
                           random_state=42)

# باینری کردن کلاس‌ها برای OvR
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# آموزش مدل
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)

# رسم ROC برای هر کلاس
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[i][:, 1])
    roc_auc[i] = auc(fpr[i], tpr[i])

# رسم نمودار
plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("'ROC Curve - Multi-Class'")
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve,accuracy_score,confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

x,y = make_classification(n_samples = 1000,n_features = 20,n_classes = 3,
n_informative = 10,weights = [0.6,0.3,0.1],random_state = 42)

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.3,random_state = 42)

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred_proba = model.predict_proba(x_test)[:,2]
y_true = (y_test == 2).astype(int)

precision, recall,threshold = precision_recall_curve(y_true,y_pred_proba)

f1_score = 2 * (precision * recall) / (precision + recall)
best_index = np.argmax(f1_score)
best_threshold = threshold[best_index]
y_pred_custom = (y_pred_proba >= best_threshold).astype(int)
acc = accuracy_score(y_test, y_pred_custom)
cm = confusion_matrix(y_test, y_pred_custom)


print("Best Threshold:",best_threshold)
print("Precison:",precision[best_index])
print("Recall:",recall[best_index])
print("F1 Score:",f1_score[best_index])
print("Accuracy:", acc)
print("Confusion Matrix:", cm)


plt.figure(figsize= (8,5))
plt.plot(threshold,precision[:-1],label= "Precision",color = "blue")
plt.plot(threshold,recall[:-1], label = "recall",color = "orange")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision vs recall at Different Threshold (Class 2)")
plt.legend()
plt.grid(True) 
plt.show()

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve,f1_score
import matplotlib.pyplot as plt
import numpy as np

x,y = make_classification(n_samples=1000,n_features = 10,n_classes=3,
                          random_state=42,n_informative=5,weights=[0.7,0.2,0.1])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_proba = model.predict_proba(x_test)[:,2]
y_true = (y_test == 2).astype(int)

precision,recall,thrshold = precision_recall_curve(y_true,y_proba)

f1_scores = []
for t in thrshold:
    y_pred = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_true,y_pred))

plt.figure(figsize = (8,5))
plt.plot(thrshold, f1_scores[:-1], label = 'F1_score', color = 'green')
plt.xlabel("thrshold")
plt.ylabel('F1 Score')
plt.title('F1 vs Threshold (class 2)')
plt.legend()
plt.grid(True)
plt.show()


best_index = np.argmax(f1_scores)
best_threshold = thrshold[best_index]
best_f1 = f1_scores[best_index]
print(f"Best thrshold for class 2: {best_threshold:.2f} by F1 = {best_f1}")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np

# ساخت دیتاست
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3,
                           random_state=42, n_informative=5, weights=[0.7,0.2,0.1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# آموزش مدل
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# احتمال تعلق به کلاس 2
y_proba = model.predict_proba(X_test)[:, 2]
y_true = (y_test == 2).astype(int)

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

# محاسبه F1 برای هر Threshold
f1_scores = []
for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_true, y_pred))

# رسم نمودار
plt.figure(figsize=(8,5))
plt.plot(thresholds, f1_scores, label='F1-Score', color='green')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold (Class 2)")
plt.legend()
plt.grid(True)
plt.show()

# پیدا کردن بهترین Threshold
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]
print(f"🔍 Best threshold for class 2: {best_threshold:.2f} with F1 = {best_f1:.2f}")

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


x,y = make_classification(n_samples=1000,n_classes=3,n_features=10,n_informative=5,
                          random_state=42,weights=[1.0,0.0,0.0])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
disp.plot(cmap= plt.cm.Blues)
plt.title("Confusion Matrix - Day 26")
plt.show()
print(disp)

from sklearn.metrics import precision_score,recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x,y = make_classification(n_samples=1000,random_state=42,n_classes=3,n_features=10,n_informative=5
                          ,weights = [0.8,0.15,0.05])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

precision_micro = precision_score(y_test,y_pred,average="micro")
recall_micro = recall_score(y_test,y_pred,average='micro')
f1_micro = f1_score(y_test,y_pred,average='micro')

precision_macro = precision_score(y_test,y_pred, average='macro')
recall_macro = recall_score(y_test,y_pred,average="macro")
f1_macro = f1_score(y_test,y_pred,average= 'macro')

print("Micro Averaging:")
print(f"Precision: {precision_micro:.2f} Recall: {recall_micro:.2f} F1: {f1_micro:.2f}")

print("Macro Averaging:")
print(f"Precision: {precision_macro:.2f} Recall: {recall_macro:.2f} F1: {f1_macro:.2f}")

from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

x,y = make_classification(n_classes= 3,n_samples= 1000, random_state= 42,n_informative= 5, n_features= 10,
                          weights= [0.8,0.15,0.05])

model = RandomForestClassifier(random_state=42)
cv = StratifiedKFold(n_splits= 5, shuffle = True, random_state= 42)

f1_macro  = cross_val_score(model, x,y, cv = cv, scoring = make_scorer(f1_score, average = 'macro'))
precision_macro = cross_val_score(model, x, y, cv = cv, scoring= make_scorer(precision_score, average = 'macro'))
recall_macro = cross_val_score(model, x, y, cv = cv, scoring=make_scorer(recall_score, average = 'macro'))

print("Cross-validation Result(Macro Averaging):")
print(f"F1: {np.mean(f1_macro):.2f} ± {np.std(f1_macro):.2f}")
print(f"Precision: {np.mean(precision_macro):.2f} ± {np.std(precision_macro):.2f}")
print(f"Recall: {np.mean(recall_macro):.2f} ± {np.std(recall_macro):.2f}")


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

x,y = make_classification(n_classes=3,n_samples=1000,n_features=10,n_informative=5,
                          random_state=42, weights=[0.8,0.15,0.5])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model_normal = RandomForestClassifier(random_state=42)
model_normal.fit(x_train,y_train)
y_pred_normal = model_normal.predict(x_test)

print("With out class Weight:")
print(classification_report(y_test,y_pred_normal))


model_weighted = RandomForestClassifier(class_weight='balanced',random_state=42)
model_weighted.fit(x_train,y_train)
y_pred_weight = model_weighted.predict(x_test)

print("With class Weight:")
print(classification_report(y_test,y_pred_weight))



from sklearn.metrics import roc_curve,auc, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('diabetes.csv')

x = data.drop('Outcome',axis = 1)
y = data['Outcome']

x_train,x_test,y_train,y_test =train_test_split(x,y,
                                                test_size=0.3,random_state=42)

model = LogisticRegression(class_weight='balanced',max_iter = 1000)
model.fit(x_train,y_train)

y_pred_proba = model.predict_proba(x_test)[:,1]


fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)
roc_auc = auc(fpr,tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color = 'blue', label = f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate (Recall)') 
plt.title('ROC Curve - Diabetes Dataset') 
plt.legend(loc="lower right") 
plt.show()

thresholds = 0.3
y_pred = (y_pred_proba >= thresholds).astype(int)

print("Class Report:")
print(classification_report(y_test,y_pred))


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

df = pd.read_csv('diabetes.csv')

x = df.drop('Outcome', axis = 1)
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.3,random_state=42)
model = RandomForestClassifier(max_depth=7,random_state=42,n_estimators=10)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print(f'Accuracy: {accuracy_score(y_pred,y_test):.2f}')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,precision_score,f1_score,recall_score,make_scorer
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score

data = pd.read_csv('diabetes.csv')

x = data.drop("Outcome",axis = 1)
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y, 
                                                 test_size=0.3,random_state=42)

model = RandomForestClassifier(random_state=42,class_weight='balanced')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


cv = StratifiedKFold(random_state=42,shuffle = True,n_splits=5)
precision_macro = cross_val_score(model,x,y,cv = cv,scoring=make_scorer(precision_score,average = 'macro'))
recall_macro = cross_val_score(model,x,y, cv=cv, scoring=make_scorer(recall_score,average = 'macro'))
f1_macro = cross_val_score(model,x,y, cv = cv, scoring=make_scorer(f1_score,average = 'macro'))

import numpy as np



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(32,input_dim = x_train.shape[1],activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor= 'var_loss',patience = 5,restore_best_weights = True,mode = 'max')

history = model.fit(x_train,y_train,epochs = 100,batch_size = 10,validation_data = (x_test,y_test),
                    callbacks = [early_stopping])

loss, accuracy = model.evaluate(x_test,y_test)

print("Machin Learning Accuracy")
print("Class report:", classification_report(y_test,y_pred))
print(f"F1.macro mean: {np.mean(f1_macro):.2f} Macro std: {np.std(f1_macro):.2f}")
print(f"Recall.marco mean {np.mean(recall_macro):.2f} Macro std: {np.std(recall_macro):.2f}")
print(f"Presicion: macro mean {np.mean(precision_macro):.2f} Macro std: {np.std(precision_macro):.2f} ")

print(f" Neural Network Accuracy : {accuracy:.2f}")


person = {"Name": 'ALi',"age":25}

name = person["Name"]
age = person['age']

name = person.get("Name")

country = person.get("Country","Iran")

for per in person.items():
    print(f"{per}")


