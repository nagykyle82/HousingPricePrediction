# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:07:26 2021

@author: nagyk
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Count up missing values for each column
def missing_val_count(df):
    count = df.isnull().sum()
    print(count[count > 0])
    
#Read in data from .csv files
def load_data():
    # Read data
    df_train = pd.read_csv("train.csv", index_col="Id")
    df_test = pd.read_csv("test.csv", index_col="Id")
    
    # Preprocessing
    num_cols_train = [col for col in df_train.columns if df_train[col].dtype in ['int64','float64']]
    num_cols_test = [col for col in df_test.columns if df_test[col].dtype in ['int64','float64']]
    df_train = df_train[num_cols_train]
    df_test = df_test[num_cols_test]
    MissGrgYearNeigh_train = df_train['GarageYrBlt'].median(skipna=True)
    MissGrgYearNeigh_test = df_test['GarageYrBlt'].median(skipna=True)
    df_train = rename(df_train)
    df_test = rename(df_test)
    df_train = df_train.fillna(value=MissGrgYearNeigh_train)
    df_test = df_test.fillna(value=MissGrgYearNeigh_test)
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)
    
    # Drop columns because of >250 missing entries or categorical (numerical)
    drop_cols = ['LotFrontage', 'MSSubClass', 'OverallQual', 'OverallCond']
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
'''
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

# Examine relationships between features as well as target
df_train, df_test = load_data()
cols1 = ['GarageArea', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF', 'SalePrice']
cols2 = ['LotArea', 'GarageCars', 'GarageYrBlt', 'FirstFlrSF', 'SalePrice']
cols3 = ['YearRemodAdd', 'FullBath', 'TotRmsAbvGrd', 'OpenPorchSF', 'SalePrice']
cols4 = ['Fireplaces', 'SecondFlrSF', 'BsmtFinSF1', 'BsmtUnfSF', 'SalePrice']
cols5 = ['WoodDeckSF', 'MasVnrArea', 'HalfBath', 'BedroomAbvGr', 'SalePrice']
cols6 = ['KitchenAbvGr', 'BsmtFullBath', 'EnclosedPorch', 'ScreenPorch', 'SalePrice']
cols7 = ['BsmtHalfBath', 'LowQualFinSF', 'YrSold', 'BsmtFinSF2', 'SalePrice']
cols8 = ['ThreeSsnporch', 'MiscVal', 'PoolArea', 'MoSold', 'SalePrice']
cols = cols2
plt.figure()
sns.pairplot(df_train[cols], corner=True)
plt.tight_layout()
plt.show()
'''
# Load data, parse out feature set and target set, divide into training and validation sets
X, X_test = load_data()
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75,
                                                      test_size=0.25, random_state=0)
model = XGBRegressor(n_estimators=100, learning_rate=0.113)
'''
#Identify most important features
mi_scores = make_mi_scores(X, y)
print(mi_scores.head(50))
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(40))

# Test various regressors
regressors = [
    LinearRegression(),
    Lasso(),
    ElasticNet(),
    Ridge(),
    LinearSVR(),
    SVR(),
    XGBRegressor(n_estimators=100, learning_rate=0.1),
    RandomForestRegressor()]

log_cols=["Regressor", "MAE", "R2"]
log = pd.DataFrame(columns=log_cols)

for reg in regressors:
    reg.fit(X_train, y_train)
    name = reg.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    y_train_pred = reg.predict(X_train)
    mae = mean_absolute_error(y_train, y_train_pred)
    print("MAE: {}".format(mae))
    
    r2 = r2_score(y_train, y_train_pred)
    print("R2: %.3f" % r2)
    
    log_entry = pd.DataFrame([[name, mae, r2]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

# Validation Curve
param_range = list(range(50,450,50))
train_scores, test_scores = validation_curve(estimator=model,
                                             X=X_train,
                                             y=y_train,
                                             param_name='n_estimators',
                                             param_range=param_range,
                                             cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', label='training')
plt.plot(param_range, test_mean, color='red', marker='s', label='validation')
plt.legend(loc='lower right')
plt.show

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(estimator=model,
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
'''
# Residuals plot
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
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
