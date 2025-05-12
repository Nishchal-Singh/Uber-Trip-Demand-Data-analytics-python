import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

data_files = [
    'uber-raw-data-apr14.csv',
    'uber-raw-data-may14.csv',
    'uber-raw-data-jun14.csv',
    'uber-raw-data-jul14.csv',
    'uber-raw-data-aug14.csv',
    'uber-raw-data-sep14.csv'
]
dfs = [pd.read_csv(f) for f in data_files]
uber_df = pd.concat(dfs)

uber_df['Date/Time'] = pd.to_datetime(uber_df['Date/Time'])
uber_df.rename(columns={'Date/Time': 'Date'}, inplace=True)
uber_df.set_index('Date', inplace=True)
uber_df.sort_index(inplace=True)

hourly_data = uber_df['Base'].resample('H').count().reset_index()
hourly_data.columns = ['Date', 'Count']
hourly_data.set_index('Date', inplace=True)

plt.figure(figsize=(20,6))
plt.plot(hourly_data['Count'], color='darkblue')
plt.title('Hourly Uber Trip Count (2014)')
plt.show()

decomposition = seasonal_decompose(hourly_data['Count'], model='add', period=24)
decomposition.plot()
plt.show()

cutoff = '2014-09-15'
train = hourly_data.loc[:cutoff]
test = hourly_data.loc[cutoff:]

def create_lagged_features(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window_size = 24
X_train, y_train = create_lagged_features(train['Count'].values, window_size)

test_input = np.concatenate([train['Count'].values[-window_size:], test['Count'].values])
X_test, y_test = create_lagged_features(test_input, window_size)

tscv = TimeSeriesSplit(n_splits=5)
seed = 42

xgb_params = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6],
    'learning_rate': [0.1],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
xgb_cv = GridSearchCV(xgb_model, xgb_params, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
xgb_cv.fit(X_train, y_train)
xgb_pred = xgb_cv.best_estimator_.predict(X_test)

rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None]
}
rf_model = RandomForestRegressor(random_state=seed)
rf_cv = GridSearchCV(rf_model, rf_params, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
rf_cv.fit(X_train, y_train)
rf_pred = rf_cv.best_estimator_.predict(X_test)

gbr_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}
gbr_model = GradientBoostingRegressor(random_state=seed)
gbr_cv = GridSearchCV(gbr_model, gbr_params, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
gbr_cv.fit(X_train, y_train)
gbr_pred = gbr_cv.best_estimator_.predict(X_test)

xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred)
rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_pred)

print(f"XGBoost MAPE: {xgb_mape:.2%}")
print(f"Random Forest MAPE: {rf_mape:.2%}")
print(f"GBRT MAPE: {gbr_mape:.2%}")

weights = np.array([1/xgb_mape, 1/rf_mape, 1/gbr_mape])
weights /= weights.sum()
ensemble_pred = weights[0]*xgb_pred + weights[1]*rf_pred + weights[2]*gbr_pred
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred)

print(f"Ensemble MAPE: {ensemble_mape:.2%}")

plt.figure(figsize=(18,6))
plt.plot(test.index[window_size:], y_test, label='Actual', color='gray')
plt.plot(test.index[window_size:], xgb_pred, '--', label='XGBoost', color='red')
plt.plot(test.index[window_size:], rf_pred, '--', label='Random Forest', color='green')
plt.plot(test.index[window_size:], gbr_pred, '--', label='GBRT', color='orange')
plt.plot(test.index[window_size:], ensemble_pred, '-', label='Ensemble', color='purple')
plt.legend()
plt.title('Uber Trips Forecasting: Actual vs Predictions')
plt.show()
