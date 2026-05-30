import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def train(df):
    
    #Define Target and features without polution
    target = df['Survived']
    features = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket']).copy()
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    #Start regression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    return accuracy_score(Y_test, Y_pred)

def forest_train(df):

    #Define Target and features without polution
    target = df['Survived']
    features = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket']).copy()
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    #Start Random forest
    forest = RandomForestClassifier(n_estimators=500, random_state=42, oob_score=True, max_depth=15, min_samples_split=10)


    forest.fit(X_train, Y_train)

    Y_pred = forest.predict(X_test)
    
    return accuracy_score(Y_test, Y_pred)


