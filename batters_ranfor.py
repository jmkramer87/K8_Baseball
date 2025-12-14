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
df = pd.read_csv('Batters_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

y = df[['r_run', 'batting_avg', 'b_rbi', 'r_total_stolen_base', 'home_run']]
X = df.drop(columns=['hit', 'single', 'double', 'triple', 'home_run', 'player_id', 'b_rbi', 'r_total_stolen_base', 'batting_avg', 'on_base_percent', 'r_run', 'xba', 'xslg', 'xobp'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

f = open("log-ranfor.txt", "a")

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"Linear R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Linear RMSE: {RMSE}\n")

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

f.write("Successful run!")
f.write(f"Total time: {time.time() - start_time}")

f.close()

