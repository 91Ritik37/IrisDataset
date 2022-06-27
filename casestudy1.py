#Harijan Ritik
import pandas as pd

df =pd.read_csv("C:/Users/CC-078/Downloads/iris.csv")
from  sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
naive=MultinomialNB()
rfc=RandomForestClassifier(random_state=1)
SVM=svm.SVC()
GBC=GradientBoostingClassifier(n_estimators=10)
DTC=DecisionTreeClassifier(random_state=0)
neural=MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
logr=LogisticRegression()


X=df.drop("species" ,axis=1)
y=df["species"]

X_train ,X_test ,y_train,y_test =train_test_split(X,y,random_state=0,test_size=0.3)
#print(X_train)
#print(X_test)
#print(y_test)
#abcd=print(y_train)

logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

neural.fit(X_train,y_train)
y_neural_pred=neural.predict(X_test)

DTC.fit(X_train,y_train)
y_DTC_pred=DTC.predict(X_test)

GBC.fit(X_train,y_train)
y_GBC_pred= GBC.predict(X_test)

SVM.fit(X_train,y_train)
y_SVM_pred=SVM.predict(X_test)

rfc.fit(X_train,y_train)
y_rfc_pred=rfc.predict(X_test)

naive.fit(X_train,y_train)
y_pred_naive=naive.predict(X_test)

print("logicstics:")
print ( accuracy_score(y_test,y_pred))
print("neural_network")
print(accuracy_score(y_test,y_neural_pred))
print("DEcisin treee")
print(accuracy_score(y_test,y_DTC_pred))
print("Gradiant boosting")
print(accuracy_score(y_test,y_GBC_pred))
print("Svm")
print(accuracy_score(y_test,y_SVM_pred))
print("Random forest")
print(accuracy_score(y_test,y_rfc_pred))
print("Naive bais")
print(accuracy_score(y_test,y_pred_naive))


""""
logicstics:
0.9777777777777777
neural_network
0.24444444444444444
DEcisin treee
0.9777777777777777
Gradiant boosting
0.9777777777777777
Svm
0.9777777777777777
Random forest
0.9777777777777777
Naive bais
0.6
"""