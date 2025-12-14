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

f.write("Successful run!")
f.write(f"Total time: {time.time() - start_time}")

f.close()

