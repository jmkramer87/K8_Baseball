import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

#Categories (pitchers): p_win, strikeout, p_hold, p_save, p_era, whip
#Categories (hitters): hit, batting_avg, b_rbi, r_total_stolen_base, home runs
df = pd.read_csv('Batters_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

y = df[['hit', 'batting_avg', 'b_rbi', 'r_total_stolen_base', 'home_run']]
X = df.drop(columns=['hit', 'single', 'double', 'triple', 'home_run', 'player_id', 'b_rbi', 'r_total_stolen_base', 'batting_avg', 'on_base_percent', 'r_run', 'xba', 'xslg', 'xobp', 'xbadiff'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

f = open("logs-jkd22.txt", "a")

f.write("Starting Regression Analysis\n")

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

f.write(f"Linear R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Linear RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

#GradientBoost
param_grid = {
    'pca__n_components': range(5, 60, 1),
    'regression__max_depth': range(3, 20, 1),
    'regression__learning_rate': [0.01, 0.1, 0.2],
    'regression__subsample': [x / 10.0 for x in range(6, 10, 1)],
    'regression__min_samples_split': range(2, 20, 1),
    'regression__min_samples_leaf': [x / 10.0 for x in range(1, 10, 1)]
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", GradientBoostingRegresor(n_estimators=100))
])

grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

predictions = grid_search.predict(X_test)

f.write(f"GradientBoost best parameters: {grid_search.best_params_}\n")
f.write(f"GradientBoost best score: {grid_search.best_score_}\n")
f.write(f"GradientBoost R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"GradientBoost RMSE: {root_mean_squared_error(y_test, predictions)}\n")

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

f.write(f"DecisionTree best parameters: {grid_search.best_params_}\n")
f.write(f"DecisionTree best score: {grid_search.best_score_}\n")
f.write(f"DecisionTree R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"DecisionTree RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

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

f.write(f"RandomForest best parameters: {grid_search.best_params_}\n")
f.write(f"RandomForest best score: {grid_search.best_score_}\n")
f.write(f"RandomForest R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"RandomForest RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

#Lasso
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__alpha': [x / 100.0 for x in range(5, 20, 1)],
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

f.write(f"Lasso best parameters: {grid_search.best_params_}\n")
f.write(f"Lasso best score: {grid_search.best_score_}\n")
f.write(f"Lasso R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Lasso RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

#Ridge
param_grid = {
    'pca__n_components': range(10, 60, 1),
    'regression__alpha': [x / 100.0 for x in range(5, 20, 1)],
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

f.write(f"Ridge best parameters: {grid_search.best_params_}\n")
f.write(f"Ridge best score: {grid_search.best_score_}\n")
f.write(f"Ridge R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Ridge RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

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

f.write(f"SVR best parameters: {grid_search.best_params_}\n")
f.write(f"SVR best score: {grid_search.best_score_}\n")
f.write(f"SVR R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"SVR RMSE: {math.sqrt(mean_squared_error(y_test, predictions))}\n")

f.write("Successful run!")

f.close()

