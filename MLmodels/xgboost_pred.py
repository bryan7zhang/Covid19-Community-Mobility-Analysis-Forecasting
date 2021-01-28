import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost
import matplotlib.pyplot as plt

import dataselector

# region = 'Canada'

cols =['days_post_first_lockdown', 'day_of_week',
       'stringency_index', 'driving', 'transit', 'walking',
       'retail_and_recreation', 'grocery_and_pharmacy', 'parks',
       'transit_stations','workplaces', 'residential']

# data = pd.read_csv(f"../DataPreprocessing/data_by_regions/{region}-2020-12-01.csv")

data, region = dataselector.chooseDataset()

X = data[cols]
Y = data['numtoday']

# train model - dataset split
# to predict last 14 days, shuffle = False, test_size=0.056
train_features, test_features, train_labels, test_labels = \
       train_test_split(X, Y, test_size=0.056, shuffle=False, random_state=42)

print('region:', region)
print('Start grid search, please wait ... ')
mae = float('+inf')
best_lr = 0
best_d = 0

for i in range(3, 15):
    for j in np.arange(0.01, 0.5, 0.01):
        xgbrgr = xgboost.XGBRegressor(max_depth=i,
                                      n_estimators=1000,
                                      learning_rate=j,
                                      eval_metric='mae',
                                      objective='reg:squarederror',
                                      )
        xgbrgr.fit(train_features, train_labels)
        y_pred = xgbrgr.predict(test_features)
        mae_ = mean_absolute_error(y_pred, test_labels)
        if mae > mae_:
            mae = mae_
            best_d = i
            best_lr = j

print('region: ', region)
print('lowest mae', mae, 'best learning rate', best_lr, 'best max depth', best_d)

xgbrgr = xgboost.XGBRegressor(max_depth=best_d,
                              n_estimators=1000,
                              learning_rate=best_lr,
                              eval_metric='mae',
                              objective='reg:squarederror',
                              # objective='reg:linear'
                              )

xgbrgr.fit(train_features, train_labels)
y_pred = xgbrgr.predict(test_features)

feature_importance = xgbrgr.get_booster().get_score(importance_type="weight")
feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1]))

print("rmse ", sqrt(mean_squared_error(y_pred, test_labels, squared=False)))
print("mae ", mean_absolute_error(y_pred, test_labels))
print("percentage error: ", mean_absolute_error(y_pred, test_labels)/(test_labels.mean()))


fig, ax = plt.subplots()

y_pos = np.arange(len(feature_importance))
print(feature_importance)
ax.barh(y_pos, feature_importance.values(), align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_importance.keys())
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature performance')
ax.set_title(f'Weights of importance of different features\n'
             f'in the data from {region}')
plt.show()

# converting predictions to dataframe
pred = pd.DataFrame(data=y_pred,
                    index=range(len(y_pred)))
pred.index = test_labels.index

plt.plot(train_labels, label='train')
plt.plot(test_labels, label='test')
plt.plot(pred, label='pred')

plt.xlabel('days after first lockdown on Mar.11')
plt.ylabel('number of cases')
plt.title(f'Time series forecasting with XGBoost for daily COVID-19 cases\n'
          f'for the dataset of {region}')
plt.legend()
plt.show()
