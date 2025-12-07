import pandas as pd
import math
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
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
df = pd.read_csv('Pitchers_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

y = df[['p_save', 'p_win', 'strikeout', 'p_hold', 'p_era', 'whip']]
X = df.drop(columns=['player_id', 'p_save', 'p_win', 'strikeout', 'p_hold', 'p_era', 'whip', 'k_percent', 'p_called_strike', 'xba', 'xslg', 'xwoba'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

f = open("logs-pitchers.txt", "w")

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

f.write(f"Linear R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Linear RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"Linear Training Accuracy : {metrics.accuracy_score(y_train, model.predict(X_train))*100)}\n")
f.write(f"Linear Testing Accuracy: {metrics.accuracy_score(y_test, model.predict(X_test))*100)}\n")

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

gs_gb = GridSearchCV(pipe, param_grid, cv=5)
gs_gb.fit(X_train, y_train)

predictions = gs_gb.predict(X_test)

f.write(f"GradientBoost best parameters: {gs_gb.best_params_}\n")
f.write(f"GradientBoost best score: {gs_gb.best_score_}\n")
f.write(f"GradientBoost R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"GradientBoost RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"GradientBoost Training Accuracy : {metrics.accuracy_score(y_train, gs_gb.predict(X_train))*100)}\n")
f.write(f"GradientBoost Testing Accuracy: {metrics.accuracy_score(y_test, gs_gb.predict(X_test))*100)}\n")

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

gs_dt = GridSearchCV(pipe, param_grid, cv=5)
gs_dt.fit(X_train, y_train)

predictions = gs_dt.predict(X_test)

f.write(f"DecisionTree best parameters: {gs_dt.best_params_}\n")
f.write(f"DecisionTree best score: {gs_dt.best_score_}\n")
f.write(f"DecisionTree R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"DecisionTree RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"DecisionTree Training Accuracy : {metrics.accuracy_score(y_train, gs_dt.predict(X_train))*100)}\n")
f.write(f"DecisionTree Testing Accuracy: {metrics.accuracy_score(y_test, gs_dt.predict(X_test))*100)}\n")

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

gs_rf = GridSearchCV(pipe, param_grid, cv=5)
gs_rf.fit(X_train, y_train)

predictions = gs_rf.predict(X_test)

f.write(f"RandomForest best parameters: {gs_rf.best_params_}\n")
f.write(f"RandomForest best score: {gs_rf.best_score_}\n")
f.write(f"RandomForest R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"RandomForest RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"RandomForest Training Accuracy : {metrics.accuracy_score(y_train, gs_rf.predict(X_train))*100)}\n")
f.write(f"RandomForest Testing Accuracy: {metrics.accuracy_score(y_test, gs_rf.predict(X_test))*100)}\n")

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

gs_lasso = GridSearchCV(pipe, param_grid, cv=5)
gs_lasso.fit(X_train, y_train)

predictions = gs_lasso.predict(X_test)

f.write(f"Lasso best parameters: {gs_lasso.best_params_}\n")
f.write(f"Lasso best score: {gs_lasso.best_score_}\n")
f.write(f"Lasso R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Lasso RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"Lasso Training Accuracy : {metrics.accuracy_score(y_train, gs_lasso.predict(X_train))*100)}\n")
f.write(f"Lasso Testing Accuracy: {metrics.accuracy_score(y_test, gs_lasso.predict(X_test))*100)}\n")

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

gs_ridge = GridSearchCV(pipe, param_grid, cv=5)
gs_ridge.fit(X_train, y_train)

predictions = gs_ridge.predict(X_test)

f.write(f"Ridge best parameters: {gs_ridge.best_params_}\n")
f.write(f"Ridge best score: {gs_ridge.best_score_}\n")
f.write(f"Ridge R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Ridge RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"Ridge Training Accuracy : {metrics.accuracy_score(y_train, gs_ridge.predict(X_train))*100)}\n")
f.write(f"Ridge Testing Accuracy: {metrics.accuracy_score(y_test, gs_ridge.predict(X_test))*100)}\n")

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

gs_svr = GridSearchCV(pipe, param_grid, cv=5)
gs_svr.fit(X_train, y_train)

predictions = gs_svr.predict(X_test)

f.write(f"SVR best parameters: {gs_svr.best_params_}\n")
f.write(f"SVR best score: {gs_svr.best_score_}\n")
f.write(f"SVR R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"SVR RMSE: {root_mean_squared_error(y_test, predictions)}\n")
f.write(f"SVR Training Accuracy : {metrics.accuracy_score(y_train, gs_svr.predict(X_train))*100)}\n")
f.write(f"SVR Testing Accuracy: {metrics.accuracy_score(y_test, gs_svr.predict(X_test))*100)}\n")

f.write("Successful run!")

f.close()

