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

f = open("logp-kn.txt", "w")

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

