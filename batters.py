import pandas as pd
import math
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
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
from sklearn.neighbors import KNeighborsRegressor
import time

start_time = time.time()

#Categories (pitchers): p_win, strikeout, p_hold, p_save, p_era, whip
#Categories (hitters): r_run, batting_avg, b_rbi, r_total_stolen_base, home_run
df = pd.read_csv('~\\Downloads\\Batters_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

y = df[['r_run', 'batting_avg', 'b_rbi', 'r_total_stolen_base', 'home_run']]
X = df.drop(columns=['hit', 'single', 'double', 'triple', 'home_run', 'player_id', 'b_rbi', 'r_total_stolen_base', 'batting_avg', 'on_base_percent', 'r_run', 'xba', 'xslg', 'xobp'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

f = open("logs-batters.txt", "a")

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"Linear R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Linear RMSE: {RMSE}\n")

#GradientBoost
param_grid = {
    'pca__n_components': range(5, 50, 5),
    'regression__max_depth': range(3, 20, 1),
    'regression__learning_rate': [0.01, 0.1, 0.2],
    'regression__subsample': [x / 10.0 for x in range(6, 10, 1)],
    'regression__min_samples_split': range(2, 20, 1),
    'regression__min_samples_leaf': [x / 10.0 for x in range(1, 10, 1)]
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", GradientBoostingRegressor())
])

gs_gb = GridSearchCV(pipe, param_grid, cv=5)
MultiOutputRegressor(gs_gb).fit(X_train, y_train)

predictions = gs_gb.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"GradientBoost best parameters: {gs_gb.best_params_}\n")
f.write(f"GradientBoost best score (R2): {gs_gb.best_score_}\n")
f.write(f"GradientBoost Individual R2: {r2_score(y_test, predictions, multioutput='raw_values')*100}\n")
f.write(f"GradientBoost RMSE: {RMSE}\n")

#Decision Tree
param_grid = {
    'pca__n_components': range(10, 50, 5),
    'regression__max_depth': range(10, 30, 2),
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
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"DecisionTree best parameters: {gs_dt.best_params_}\n")
f.write(f"DecisionTree best score (overall R2): {gs_dt.best_score_}\n")
f.write(f"DecisionTree R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"DecisionTree RMSE: {RMSE}\n")

#Random Forest
param_grid = {
    'pca__n_components': range(10, 60, 5),
    'regression__max_depth': range(10, 30, 2),
    'regression__min_samples_split': range(2, 10, 1),
    'regression__min_samples_leaf': range(1, 8, 1),
    'regression__n_estimators': range(50, 200, 25)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", RandomForestRegressor())
])

gs_rf = GridSearchCV(pipe, param_grid, cv=5)
MultiOutputRegressor(gs_rf).fit(X_train, y_train)

predictions = gs_rf.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"RandomForest best parameters: {gs_rf.best_params_}\n")
f.write(f"RandomForest best score: {gs_rf.best_score_}\n")
f.write(f"RandomForest R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"RandomForest RMSE: {RMSE}\n")

#Lasso
param_grid = {
    'pca__n_components': range(10, 50, 5),
    'regression__alpha': [x / 100.0 for x in range(5, 20, 1)],
    'regression__max_iter': range(500, 2000, 250)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", MultiTaskLasso())
])

gs_lasso = GridSearchCV(pipe, param_grid, cv=5)
gs_lasso.fit(X_train, y_train)

predictions = gs_lasso.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"Lasso best parameters: {gs_lasso.best_params_}\n")
f.write(f"Lasso best score: {gs_lasso.best_score_}\n")
f.write(f"Lasso R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Lasso RMSE: {RMSE}\n")

#Ridge
param_grid = {
    'pca__n_components': range(10, 50, 5),
    'regression__alpha': [x / 100.0 for x in range(5, 20, 1)],
    'regression__max_iter': range(500, 2000, 250)
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", Ridge())
])

gs_ridge = GridSearchCV(pipe, param_grid, cv=5)
gs_ridge.fit(X_train, y_train)

predictions = gs_ridge.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"Ridge best parameters: {gs_ridge.best_params_}\n")
f.write(f"Ridge best score: {gs_ridge.best_score_}\n")
f.write(f"Ridge R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Ridge RMSE: {RMSE}\n")

#SVR
param_grid = {
    'pca__n_components': range(10, 50, 5),
    'regression__n_neighbors': range(5, 15, 1),
    'regression__weights': ['uniform', 'distance']
}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("regression", KNeighborsRegressor())
])

gs_kn = GridSearchCV(pipe, param_grid, cv=5)
gs_kn.fit(X_train, y_train)

predictions = gs_kn.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"KNeighbors best parameters: {gs_kn.best_params_}\n")
f.write(f"KNeighbors best score: {gs_kn.best_score_}\n")
f.write(f"KNeighbors R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"KNeighbors RMSE: {RMSE}\n")

f.write("Successful run!")
f.write(f"Total time: {time.time() - start_time}")

f.close()

