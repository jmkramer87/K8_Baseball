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

f = open("log-gradboost.txt", "a")

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

f.write("Successful run!")
f.write(f"Total time: {time.time() - start_time}")

f.close()

