import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    
    #Remove Useless Columns
    df = df.drop(columns=['Cabin', 'Fare', 'Embarked'])

    #Fill Columns with Median Values for consistency
    m = df['Age'].median()
    df['Age'] = df['Age'].fillna(m)

    #Encode 'Sex' Column
    df['Sex'] = df['Sex'].replace(['male', 'female'], [0,1]) 

    #Scale Columns to avoid magnitude priorization
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])

    #Create Features
    df['isAlone'] = False
    df.loc[(df['SibSp'] + df['Parch'])== 0, 'isAlone'] = True

    

    #Drop columns processed for features
    df = df.drop(columns=['SibSp', 'Parch'])

    return df