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
df = pd.read_csv('Pitchers_22-24_Avg_Clean_Nameless.csv', encoding='UTF-8')

y = df[['p_save', 'p_win', 'strikeout', 'p_hold', 'p_era', 'whip']]
X = df.drop(columns=['player_id', 'p_save', 'p_win', 'strikeout', 'p_hold', 'p_era', 'whip', 'k_percent', 'p_called_strike', 'xba', 'xslg', 'xwoba'])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

f = open("logp-ridge.txt", "w")

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
RMSE = math.sqrt(mean_squared_error(y_test, predictions))

f.write(f"Ridge best parameters: {gs_ridge.best_params_}\n")
f.write(f"Ridge best score: {gs_ridge.best_score_}\n")
f.write(f"Ridge R2: {r2_score(y_test, predictions)*100}\n")
f.write(f"Ridge RMSE: {RMSE}\n")

f.write("Successful run!")
f.write(f"Total time: {time.time() - start_time}")

f.close()

