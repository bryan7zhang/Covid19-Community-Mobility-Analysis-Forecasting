import pandas as pd
from pandas import RangeIndex
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import dataselector


def split_train_tet(data):
    # use most recent two weeks for validation
    # train_v = data[:(data.shape[0] - 14*2)]
    # valid = data[(data.shape[0] - 14*2):(data.shape[0] - 14)]
    train = data[:(data.shape[0] - 14)]
    test = data[(data.shape[0] - 14):]
    return train, test

# check stationarity
def check_stationarity(data):
    # all eigenvalues of cols should be less than one to be feasible
    print(coint_johansen(data, -1, 1).eig)


def var_train_plot(training_set, test_set, cols, features_plt):
    # fit the VAR model with train data
    model_v = VAR(endog=training_set).fit()
    # forecast the values on validation set

    # pred_y = pd.DataFrame(data=model_v.y,
    #                       index=list(range(len(model_v.y))))

    # pred_y.index = range(pred_y.shape[0])
    prediction = model_v.forecast(model_v.y, steps=len(test_set)*2)

    # converting predictions to dataframe
    pred = pd.DataFrame(data=prediction,
                        index=range(len(prediction)),
                        columns=cols)

    pred_y = pd.DataFrame(data=model_v.y,
                          index=range(len(model_v.y)),
                          columns=cols)

    for col in features_plt:
        print('rmse value for "', col, '" is: ',
              sqrt(mean_squared_error(pred[col][:len(test_set[col])], test_set[col], squared=False)))
        print('mae value for "', col, '" is:',
              mean_absolute_error(pred[col][:len(test_set[col])], test_set[col]))
        print('percentage error for "', col, '" is:',
              mean_absolute_error(pred[col][:len(test_set[col])], test_set[col])/(test_set[col].mean()))
        # print(test_set.index[0])
        # print(type(test_set.index))
        pred.index = RangeIndex(start=test_set.index[0],
                                stop=test_set.index[0] + len(pred),
                                step=1)
        plt.plot(test_set[col], label='train')
        plt.plot(test_set[col], label='test')
        plt.plot(pred[col], label='pred')
        plt.plot(pred_y[col], label='model_y')
        if 'num' in col:
            y_label = 'Number of cases'
        else:
            y_label = 'Percentage change'

        plt.xlabel('days after first lockdown on Mar.11')
        plt.ylabel(y_label)
        plt.title(f'VAR forecasting of variable "{col}"\n'
                  f'in the dataset of {region}')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    # region = 'Saskatchewan'

    # df = pd.read_csv(f'../DataPreprocessing/data_by_regions/{region}-2020-12-01.csv')
    df, region = dataselector.chooseDataset()

    # use mobility data
    cols_mobility = ['day_of_week', 'driving', 'transit', 'walking', 'retail_and_recreation', 'grocery_and_pharmacy',
                     'parks', 'transit_stations', 'workplaces', 'residential']
    # use cases data
    cols_cases = ['numtotal', 'numtoday']

    # mobility data alone
    cols = cols_mobility
    data = df[cols]
    # use historical records to train model and most recent weeks data for validation
    train, test = split_train_tet(data)
    var_train_plot(train, test, cols, cols_mobility)

    # case data + mobility data
    cols = cols_mobility + cols_cases
    data = df[cols]
    train, test = split_train_tet(data)
    var_train_plot(train, test, cols, cols_cases)