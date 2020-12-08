from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



def train_model():
    df_weekly = pd.read_csv('6_mo_weekly.csv', sep='\t')

    train = df_weekly[df_weekly['Date'] <= '2020-10-15']
    train = train.drop(['Date', 'Release', 'Y'],axis=1)
    y_train = train['Week + 1']
    X_train = train.drop('Week + 1', axis = 1)

    test = df_weekly[df_weekly['Date'] > '2020-10-15']
    test = test.drop(['Date', 'Release', 'Y'],axis=1)
    y_test = test['Week + 1']
    X_test = test.drop('Week + 1', axis=1)

    #scaler = StandardScaler()
    #train_t = scaler.fit_transform(X_train)
    #val_t = scaler.transform(X_test)

    #X_train = pd.DataFrame(train_t)
    #X_test = pd.DataFrame(val_t)

    alpha = [-5] 
    linreg = Ridge(alpha = alpha, fit_intercept=True)
    linreg.fit(X_train, y_train)

    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred,squared = False)
    train_r2 = r2_score(y_train,y_train_pred)

    test_rmse = mean_squared_error(y_test, y_test_pred, squared = False)
    test_r2 = r2_score(y_test, y_test_pred)

    return linreg
