# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:07:26 2021

@author: nagyk
"""
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Count up missing values for each column
def missing_val_count(df):
    count = df.isnull().sum()
    print(count[count > 0])
       
# Identify columns with excessively high cardinality
def cardinality(X):
    for col in nom_cols: print('{}: {}'.format(col, X[col].nunique()))
    #low_card_col = [col for col in nom_cols if X[col].nunique() < 10]
    #return low_card_col
    
#Read in data from .csv files
def load_data():
    # Read data
    df_train = pd.read_csv("train.csv", index_col="Id")
    df_test = pd.read_csv("test.csv", index_col="Id")
    # Preprocessing
    df_train = rename(df_train)
    df_test = rename(df_test)
    df_train = encode(df_train)
    df_test = encode(df_test)
    df_train = impute(df_train)
    df_test = impute(df_test)
    
    # Drop columns because of >250 missing entries
    drop_cols = ['LotFrontage','Alley','FireplaceQu','PoolQC','Fence',
                 'MiscFeature']
    df_train = df_train.drop(drop_cols, axis=1)
    df_test = df_test.drop(drop_cols, axis=1)
    return df_train, df_test

# Function to change names beginning with numbers
def rename(df):
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "ThreeSsnporch"
        }, inplace=True
        )
    return df

# Correctly identify non-numeric columns as nominal or ordinal
def encode(df):
    # Nominal categories
    for name in nom_cols:
        df[name] = df[name].astype("category")
  
    # Ordinal categories and levels
    for name, levels in ordinal_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df

# Fill in missing values
def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    imp = SimpleImputer(strategy="most_frequent")
    df[nom_cols] = imp.fit_transform(df[nom_cols])
    df[ord_cols] = imp.fit_transform(df[ord_cols])
    return df

# Identify and plot most important features
def make_mi_scores(X, y):
    for col in X.select_dtypes(['int64','float64','object','category']):
        X[col], _ = X[col].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features,
                                       random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
# The nominal categorical features
nom_cols = ['MSSubClass','MSZoning','Street','LotShape',
                'LandContour','LotConfig','LandSlope','Neighborhood',
                'Condition1','Condition2','BldgType','HouseStyle',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',
                'MasVnrType','Foundation','CentralAir','Heating',
                'GarageType','SaleType','SaleCondition'
                ]

# The ordinal categorical features
ordinal_levels = {
    'Utilities': ['ELO', 'NoSewr', 'AllPub'],#0
    'OverallQual': [3,4,5,6,7,8,9],#1
    'OverallCond': [4,5,6,7,8,9,10],#2
    'ExterQual': ['Po', 'TA', 'Gd', 'Ex'],#3
    'ExterCond': ['TA', 'Gd'],#4
    'BsmtQual': ['NA', 'Po', 'Fa', 'TA', 'Gd','Ex'],#5
    'BsmtCond': ['NA', 'Fa', 'TA', 'Gd','Ex'],#6
    'BsmtExposure': ['NA', 'No', 'Mn', 'Av', 'Gd'],#7
    'BsmtFinType1': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],#8
    'BsmtFinType2': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'GLQ'],#9
    'HeatingQC': ['Fa', 'TA', 'Gd', 'Ex'],#10
    'Electrical': ['FuseF', 'FuseA', 'SBrkr'],#11
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],#12
    'Functional': ['Sal', 'Min2', 'Min1', 'Typ'],#13
    'GarageFinish': ['NA', 'Unf','RFn', 'Fin'],#14
    'GarageQual': ['NA', 'Fa', 'TA', 'Gd'],#15
    'GarageCond': ['NA', 'Fa', 'TA', 'Gd'],#16
    'PavedDrive': ['N', 'P', 'Y'],#17
    }
ord_cols = list(ordinal_levels.keys())

# Load data, parse out feature set and target set, divide into training and validation sets
X, X_test = load_data()
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75,
                                                      test_size=0.25, random_state=0)
'''
#Identify most important features
mi_scores = make_mi_scores(X, y)
print(mi_scores.head(50))
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(40))
'''
# Identify columns by type and encode categorical data and establish model
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64','float64']]
OH_encoder = OneHotEncoder(handle_unknown='ignore')
ord_encoder = OrdinalEncoder(handle_unknown='error')
preprocessor = ColumnTransformer(
    transformers=[
        ('OH', OH_encoder, nom_cols),
        ('ord', ord_encoder, ord_cols)])
model = XGBRegressor(n_estimators=150, learning_rate=0.11, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
'''
# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=my_pipeline,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(
                                                            0.1, 1.0, 10),
                                                        cv = 5,
                                                        n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='training')
plt.plot(train_sizes, test_mean, color='red', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show

# Validation Curve
param_range = np.linspace(0.01,0.20,20)
train_scores, test_scores = validation_curve(estimator=my_pipeline,
                                             X=X_train,
                                             y=y_train,
                                             param_name='model__learning_rate',
                                             param_range=param_range,
                                             cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', label='training')
plt.plot(param_range, test_mean, color='red', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show
'''
# Residuals plot
my_pipeline.fit(X_train, y_train)
y_train_pred = my_pipeline.predict(X_train)
y_valid_pred = my_pipeline.predict(X_valid)
print('MAE train: %.3f, valid %.3f' % (
    mean_absolute_error(y_train, y_train_pred),
    mean_absolute_error(y_valid, y_valid_pred)))
print('r^2 train: %.3f, valid %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_valid, y_valid_pred)))
plt.scatter(y_train_pred, (y_train_pred - y_train), c='blue', marker='o', label='Training')
#plt.scatter(y_valid_pred, (y_valid_pred - y_valid), c='red', marker='s', label='Validation')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='lower right')
plt.show()
