import pandas as pd
import numpy as np
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

#Categories (pitchers): p_win, strikeout, p_hold, p_save, p_era, whip
#Categories (hitters): hit, batting_avg, b_rbi, r_total_stolen_base, home runs
df = pd.read_csv('~/project1/K8_Baseball/data/Batters 22-24 Average Cleaned.csv', encoding='UTF-8')

y = df[['hit', 'batting_avg', 'b_rbi', 'r_total_stolen_base', 'home_run']]
X = df.drop(columns=['hit', 'last_name, first_name', 'single', 'double', 'triple', 'home_run', 'player_id', 'b_rbi', 'r_total_stolen_base', 'batting_avg', 'on_base_percent', 'r_run', 'xba', 'xslg', 'xobp', 'xbadiff'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"Linear R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"Linear RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#XGBoost
param_grid = {
    'pca__n_components': range(5, 60, 1),
    'regression__max_depth': range(3, 9, 1),
    'regression__learning_rate': [0.01, 0.1, 0.2],
    'regression__subsample': range(0.6, 1.0, 0.1),
    'regression__colsample_bytree': range(0.6, 1.0, 0.1),
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', n_jobs=-1))
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"XGBoost best parameters: {grid_search.best_params_}", file=f)
  print(f"XGBoost best score: {grid_search.best_score_}", file=f)

  print(f"XGBoost R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"XGBoost RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#Decision Tree
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__max_depth': range(10, 30, 1),
    'regression__min_samples_split': range(2, 10, 1),
    'regression__min_samples_leaf': range(1, 8, 1)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", DecisionTreeRegressor())
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"DecisionTree best parameters: {grid_search.best_params_}", file=f)
  print(f"DecisionTree best score: {grid_search.best_score_}", file=f)

  print(f"DecisionTree R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"DecisionTree RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#Random Forest
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__max_depth': range(10, 30, 1),
    'regression__min_samples_split': range(2, 10, 1),
    'regression__min_samples_leaf': range(1, 8, 1),
    'regression__n_estimators': range(50, 200, 10)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", RandomForestRegressor())
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"RandomForest best parameters: {grid_search.best_params_}", file=f)
  print(f"RandomForest best score: {grid_search.best_score_}", file=f)

  print(f"RandomForest R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"RandomForest RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#Lasso
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__alpha': range(0.05, 0.2, 0.01),
    'regression__max_iter': range(500, 2000, 100)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", Lasso())
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"Lasso best parameters: {grid_search.best_params_}", file=f)
  print(f"Lasso best score: {grid_search.best_score_}", file=f)

  print(f"Lasso R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"Lasso RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#Ridge
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__alpha': range(0.05, 0.2, 0.01),
    'regression__max_iter': range(500, 2000, 100)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", Ridge())
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"Ridge best parameters: {grid_search.best_params_}", file=f)
  print(f"Ridge best score: {grid_search.best_score_}", file=f)

  print(f"Ridge R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"Ridge RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

#SVR
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__epsilon': [0.001, 0.01, 0.1, 1],
    'regression__C': range(1, 100, 1),
    'regression__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", SVR())
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

with open("logs-jkd22.txt", "a") as f:
  print(f"SVR best parameters: {grid_search.best_params_}", file=f)
  print(f"SVR best score: {grid_search.best_score_}", file=f)

  print(f"SVR R2: {r2_score(y_test, predictions)*100}", file=f)
  print(f"SVR RMSE: {root_mean_squared_error(y_test, predictions)}", file=f)

