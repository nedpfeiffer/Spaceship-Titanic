#!/usr/bin/python3

## IMPORT LIBRARIES
from pycaret.classification import *
import xgboost
import pandas as pd
# import shap

## INITIALIZE DATAFRAMES
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")
df_submit = pd.read_csv('./data/sample_submission.csv')

## CLEAN TRAINING DATA
df_train['Transported'] = df_train['Transported'].astype(int)
df_train['CryoSleep'] = df_train['CryoSleep'].fillna(value=0)
df_train['CryoSleep'] = df_train['CryoSleep'].astype(int)
df_train['VIP'] = df_train['VIP'].fillna(value=0)
df_train['VIP'] = df_train['VIP'].astype(int)
df_train = df_train.drop('Name', axis=1)
df_train['Age'] = df_train['Age'].fillna(value=27)
spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column in spending_columns:
    df_train[column] = df_train[column].fillna(value=0)
df_train[['Deck', 'Cabin_num', 'Side']] = df_train['Cabin'].str.split('/', expand=True)
df_train = df_train.drop('Cabin', axis=1)

## FORMAT TESTING DATA
df_test = df_test.drop('Name', axis=1)
df_test[['Deck', 'Cabin_num', 'Side']] = df_test['Cabin'].str.split('/', expand=True)
df_test = df_test.drop('Cabin', axis=1)

## INITIALIZE PYCARET
clf1 = setup(data=df_train, target='Transported',
            ignore_features=['PassengerId'],
            categorical_features=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Cabin_num', 'Side'], 
            numeric_features=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])
best = clf1.compare_models()
tuned = tune_model(best)

## FEATURE IMPORTANCE ANALYSIS
plot_model(tuned, plot="feature", save=True)

## MAKE PREDICTIONS
predictions = predict_model(tuned, data=df_test, raw_score=True)

## CLEAN THE DATA FOR SUBMISSION
transported = predictions["prediction_label"]
df_submit = df_submit.join(transported)
df_submit = df_submit.drop('Transported', axis=1)
df_submit.rename(columns={"prediction_label": "Transported"}, inplace=True)
df_submit['Transported'] = df_submit['Transported'].astype(bool)

## EXPORT TO CSV
df_submit.to_csv("./results.csv", index=False)

