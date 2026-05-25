import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    
    #Remove Useless Columns
    df = df.drop(columns='Cabin')

    #Fill Columns with Median Values for consistency
    m = df['Age'].median()
    df['Age'] = df['Age'].fillna(m)

    #Encode 'Sex' Column
    df['Sex'] = df['Sex'].replace(['male', 'female'], [0,1]) 

    #Normalize "Embarked" Column
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    df = df.drop(columns=['Embarked'])

    #Scale Columns to avoid magnitude priorization
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Fare'] = scaler.fit_transform(df[['Fare']])

    #Create Features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['isAlone'] = False
    df.loc[df['FamilySize'] == 1, 'isAlone'] = True

    #Drop columns processed for features
    df = df.drop(columns=['SibSp', 'Parch'])

    return df