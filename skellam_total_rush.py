#!/usr/bin/env python

import argparse
import collections
import datetime
import pandas as pd
import numpy as np
from numpy import *
from numpy.linalg import norm
from patsy import dmatrices
import pickle
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

data_path = "plosone_"

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--start_s", default=None, type=int, help="0 to 300")
parser.add_argument("--end_s", default=None, type=int, help="0 to 300")
parser.add_argument("--control", default=0, type=int, help="1 or 0")
parser.add_argument("--peak", default=1, type=int, help="1 or 0")
parser.add_argument("--seed_offset", default=0, type=int, help="any integer")
args = parser.parse_args()
start_iter = args.start_s
end_iter = args.end_s
control = bool(args.control)
if args.peak == 1:
    task = "peak"
else:
    task = "non-peak"

seed_offset = args.seed_offset

# this shrinking parameter can be further tuned to higher the performance.
lambda_reg = 0

MSEs = []
MSE_specials = []

for id_focused in range(start_iter, end_iter):

    seed_MSEs = {}
    general_MSEs = {}
    try:
        df_train = pd.read_pickle(data_path+'depts_2018/depts_whole_'+str(id_focused)+'.pkl')
        print('Starting station ID:', id_focused)
    except:
        continue
    df_arrivs = pd.read_pickle(data_path+'arrivs_2018/arrivs_whole_'+str(id_focused)+'.pkl')

    df_totalDepts = pd.read_pickle(data_path+'depts_2018/depts_UD_whole_'+str(id_focused)+'.pkl')
    df_totalArrivs = pd.read_pickle(data_path+'arrivs_2018/arrivs_UD_whole_'+str(id_focused)+'.pkl')

    moment_to_UDDepts = pd.read_pickle(data_path+'depts_temp_df'+'/depts_UD_dict_'+str(id_focused)+'.pkl')
    moment_to_UDArrivs = pd.read_pickle(data_path+'arrivs_temp_df'+'/arrivs_UD_dict_'+str(id_focused)+'.pkl')

    moment_to_totalDepts = {}
    for record_idx, record in df_totalDepts.iterrows():
        # date = record['date']
        hour_idx = record['hour_idx']
        day_m = record['day_m']
        month = record['month']
        rate = record['rate']
        moment_to_totalDepts[(month, day_m, hour_idx)] = rate

    moment_to_totalArrivs = {}
    for record_idx, record in df_totalArrivs.iterrows():
        # date = record['date']
        hour_idx = record['hour_idx']
        day_m = record['day_m']
        month = record['month']
        rate = record['rate']
        moment_to_totalArrivs[(month, day_m, hour_idx)] = rate

    seed_val = id_focused + seed_offset

    if task == 'peak':
        peaks = {
                16: True,
                16.5: True,
                17: True,
                17.5: True,
                18: True
                }
        quarter = '_Q3'

        moment_to_rateArrivs = {}
        for record_idx, record in df_arrivs.iterrows():
            # date = record['date']
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            rate = record['rate']
            moment_to_rateArrivs[(month, day_m, hour_idx)] = rate

        # append arrival rates to training
        rateArrivs = []
        idxes_drop = []

        for record_idx, record in df_train.iterrows():
            # date = record['date']
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            rate = moment_to_rateArrivs[(month, day_m, hour_idx)]
            rateArrivs.append(rate)
            if not (record['quarter'] == quarter and hour_idx in peaks and record['holiday'] == 0):
                idxes_drop.append(record_idx)

            if int(rate) != rate or int(record['rate']) != record['rate']:
                print('train\n', record['rate'], rate)

        se = pd.Series(rateArrivs)
        df_train['rateArrivs'] = se.values

        if idxes_drop:
            df_train.drop(idxes_drop, inplace = True)
            df_train.reset_index(inplace = True)

        print('whole set created with size', len(df_train))

        ## create X_train
        expr = """rate ~ rateArrivs + C(date) + C(hour_idx) + holiday + Clouds + cloud + Thunderstorm + Rain  + Snow \
            + wind_speed + Mist + temp \
        """

        # train
        np.random.seed(seed_val)

        # regular train & test split
        mask = np.random.rand(len(df_train)) < 0.8

        df_valid = df_train[~mask]
        df_train = df_train[mask]
        len_train = len(df_train)

        # df_train = df_train.append(df_test)    #  "y" are tuples with UD

        print('number of tuples', len_train)

        rateDepts = []
        rateArrivs = []
        for record_idx, record in df_train.iterrows():
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            dept_diff = 0
            arriv_diff = 0
            if control:
                hour_dec, hour_int = math.modf(hour_idx)
                formatted_date = (month, day_m, int(hour_int), int(bool(hour_dec)))
                if formatted_date in moment_to_UDDepts:
                    dept_diff = moment_to_UDDepts[formatted_date][0]
                if formatted_date in moment_to_UDArrivs:
                    arriv_diff = moment_to_UDArrivs[formatted_date][0]
            rateDepts.append(record['rate']-dept_diff)
            rateArrivs.append(record['rateArrivs']-arriv_diff)

        #  integer "y"
        df_train = df_train.append(df_valid)
        y_temp, X_temp = dmatrices(expr, df_train, return_type='dataframe')

        y_train, X_train = y_temp[:len_train], X_temp[:len_train]
        y_valid, X_valid = y_temp[len_train:], X_temp[len_train:]

        X_train = X_train.drop(columns=['Intercept', 'rateArrivs'])
        infos = X_train.to_numpy()

        num_features = len(infos[0])
        print('number of features: ', num_features)
        infos = np.asarray(infos)

        # y is the training label for Skellam
        diff = []
        # mu12s_actual = []
        for dept_rate, arriv_rate in zip(rateDepts, rateArrivs):
            diff.append(dept_rate-arriv_rate)

        y = diff

        def MLERegression(params):
            # int1, int2, beta1, beta2 = params[0], params[1] , params[2], params[3] # inputs are guesses at our parameters
            beta1 = params[:num_features]
            beta2 = params[num_features:2*num_features]
            int1 = params[-2]
            int2 = params[-1]
            l1 = int1 + np.dot(infos, beta1)
            l2 = int2 + np.dot(infos, beta2)
            negLL = -np.sum(stats.skellam.logpmf(y, mu1=np.exp(l1), mu2=np.exp(l2), loc=0)) + lambda_reg*(norm(beta1, 1) + norm(beta2, 1))
            return(negLL)

        x0 = np.asarray([0.4] * (2*num_features+2))
        # x0 = np.random.uniform(low=-1, high=1, size=(2*num_features+2,))

        start = datetime.datetime.now()
        x = MLERegression(x0)
        end = datetime.datetime.now()

        start = datetime.datetime.now()
        method_name = "CG"
        results = minimize(MLERegression, x0, method="CG",
                           options={'disp': True, 'maxiter': 2000})
        end = datetime.datetime.now()

        rateDepts = []
        rateArrivs = []
        for record_idx, record in df_valid.iterrows():
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            if (month, day_m, hour_idx) in moment_to_totalDepts:
                # moment_to_totalDepts: <dict, {time period with UD: total
                # demand ground truth (float number)}>
                rateDept = moment_to_totalDepts[(month, day_m, hour_idx)]
            else:
                rateDept = record['rate']
            rateDepts.append(rateDept)
            if (month, day_m, hour_idx) in moment_to_totalArrivs:
                rateArriv = moment_to_totalArrivs[(month, day_m, hour_idx)]
            else:
                rateArriv = record['rateArrivs']
            rateArrivs.append(rateArriv)

        X_valid = X_valid.drop(columns=['Intercept', 'rateArrivs'])
        infos = X_valid.to_numpy()

        diff = []
        y = []
        mu12s_actual = []
        for dept_rate, arriv_rate in zip(rateDepts, rateArrivs):
            diff.append(dept_rate-arriv_rate)
            mu12s_actual.append([dept_rate,arriv_rate])
        y = diff

        m_hat = []
        mu12s = []
        for info in infos:
            mu1_hat = np.exp(results.x[-2] + np.dot(results.x[:num_features],info))
            mu2_hat = np.exp(results.x[-1] + np.dot(results.x[num_features:2*num_features],info))
            mu12s.append([mu1_hat,mu2_hat])
            m_hat.append(mu1_hat-mu2_hat)

        mse_special = []

        for pred, actual, DA in zip(m_hat, y, mu12s_actual):
            error_val = (pred-actual)**2
            MSEs.append(error_val)
            if int(actual) != actual:
                MSE_specials.append(error_val)
                mse_special.append(error_val)

        if mse_special:
            seed_MSEs[seed_val] = np.mean(mse_special)
            print('MSE special: ', seed_MSEs[seed_val])
        try:
            general_MSEs[seed_val] = mean_squared_error(m_hat, y)
        except:
            continue
        print('MSE: ', general_MSEs[seed_val])

    else:
        peaks = {
                7: True,
                7.5: True,
                8: True,
                8.5: True,
                9: True,
                11.5: True,
                12: True,
                12.5: True,
                16: True,
                16.5: True,
                17: True,
                17.5: True,
                18: True
                }

        moment_to_rateArrivs = {}
        for record_idx, record in df_arrivs.iterrows():
            # date = record['date']
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            rate = record['rate']
            moment_to_rateArrivs[(month, day_m, hour_idx)] = rate

        # append arrival rates to training
        rateArrivs = []
        idxes_drop = []

        for record_idx, record in df_train.iterrows():
            # date = record['date']
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            rate = moment_to_rateArrivs[(month, day_m, hour_idx)]
            rateArrivs.append(rate)
            if hour_idx in peaks and record['holiday'] == 0:
                idxes_drop.append(record_idx)

            if int(rate) != rate or int(record['rate']) != record['rate']:
                print('train\n', record['rate'], rate)

        se = pd.Series(rateArrivs)
        df_train['rateArrivs'] = se.values

        if idxes_drop:
            df_train.drop(idxes_drop, inplace=True)
            df_train.reset_index(inplace=True)

        print('whole set created with size', len(df_train))

        # create X_train
        expr = """rate ~ rateArrivs + C(date) + C(hour_idx) + holiday + Clouds + cloud + Thunderstorm + Rain  + Snow \
            + wind_speed + Mist + temp \
        """

        # train
        np.random.seed(seed_val)
        mask = np.random.rand(len(df_train)) < 0.8
        df_valid = df_train[~mask]
        df_train = df_train[mask]
        len_train = len(df_train)

        # df_train = df_train.append(df_test)    #  "y" are tuples with UD

        print('number of tuples', len_train)

        rateDepts = []
        rateArrivs = []
        for record_idx, record in df_train.iterrows():
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            dept_diff = 0
            arriv_diff = 0
            if control:
                hour_dec, hour_int = math.modf(hour_idx)
                formatted_date = (month, day_m, int(hour_int), int(bool(hour_dec)))
                if formatted_date in moment_to_UDDepts:
                    dept_diff = moment_to_UDDepts[formatted_date][0]
                if formatted_date in moment_to_UDArrivs:
                    arriv_diff = moment_to_UDArrivs[formatted_date][0]
            rateDepts.append(record['rate']-dept_diff)
            rateArrivs.append(record['rateArrivs']-arriv_diff)

        # integer "y"
        df_train = df_train.append(df_valid)
        y_temp, X_temp = dmatrices(expr, df_train, return_type='dataframe')

        y_train, X_train = y_temp[:len_train], X_temp[:len_train]
        y_valid, X_valid = y_temp[len_train:], X_temp[len_train:]

        X_train = X_train.drop(columns=['Intercept', 'rateArrivs'])
        infos = X_train.to_numpy()

        num_features = len(infos[0])
        print('number of features: ', num_features)
        infos = np.asarray(infos)

        # y is the training label for Skellam
        diff = []
        y = []
        for dept_rate, arriv_rate in zip(rateDepts, rateArrivs):
            diff.append(dept_rate-arriv_rate)

        y = diff

        def MLERegression(params):
            # int1, int2, beta1, beta2 = params[0], params[1] , params[2], params[3] # inputs are guesses at our parameters
            beta1 = params[:num_features]
            beta2 = params[num_features:2*num_features]
            int1 = params[-2]
            int2 = params[-1]
            l1 = int1 + np.dot(infos,beta1)
            l2 = int2 + np.dot(infos,beta2)
            negLL = -np.sum(stats.skellam.logpmf(y, mu1=np.exp(l1), mu2=np.exp(l2), loc=0)) + lambda_reg*(norm(beta1, 1) + norm(beta2, 1))
            return(negLL)

        x0 = np.asarray([0.4] * (2*num_features+2))

        start = datetime.datetime.now()
        x = MLERegression(x0)
        end = datetime.datetime.now()

        start = datetime.datetime.now()
        method_name = "CG"
        results = minimize(MLERegression, x0, method="CG",
                           options={'disp': True, 'maxiter': 200})
        end = datetime.datetime.now()

        rateDepts = []
        rateArrivs = []
        for record_idx, record in df_valid.iterrows():
            hour_idx = record['hour_idx']
            day_m = record['day_m']
            month = record['month']
            if (month, day_m, hour_idx) in moment_to_totalDepts:
                rateDept = moment_to_totalDepts[(month, day_m, hour_idx)]
            else:
                rateDept = record['rate']
            rateDepts.append(rateDept)
            if (month, day_m, hour_idx) in moment_to_totalArrivs:
                rateArriv = moment_to_totalArrivs[(month, day_m, hour_idx)]
            else:
                rateArriv = record['rateArrivs']
            rateArrivs.append(rateArriv)

        X_valid = X_valid.drop(columns=['Intercept', 'rateArrivs'])
        infos = X_valid.to_numpy()

        diff = []
        y = []
        mu12s_actual = []
        for dept_rate, arriv_rate in zip(rateDepts, rateArrivs):
            diff.append(dept_rate-arriv_rate)
            mu12s_actual.append([dept_rate,arriv_rate])
        y = diff

        m_hat = []
        mu12s = []
        for info in infos:
            mu1_hat = np.exp(results.x[-2] + np.dot(results.x[:num_features],info))
            mu2_hat = np.exp(results.x[-1] + np.dot(results.x[num_features:2*num_features],info))
            mu12s.append([mu1_hat, mu2_hat])
            m_hat.append(mu1_hat-mu2_hat)

        mse_special = []

        for pred, actual, DA in zip(m_hat, y, mu12s_actual):
            error_val = (pred-actual)**2
            MSEs.append(error_val)
            if int(actual) != actual:
                MSE_specials.append(error_val)
                mse_special.append(error_val)

        if mse_special:
            seed_MSEs[seed_val] = np.mean(mse_special)
            print('MSE special: ', seed_MSEs[seed_val])
        try:
            general_MSEs[seed_val] = mean_squared_error(m_hat, y)
        except:
            continue
        print('MSE: ', general_MSEs[seed_val])

    if task == "non-peak":
        peak_str = "_np"
    else:
        peak_str = ''

    if control:
        control_str = "_c"
    else:
        control_str = ''

print('hour type:', task)
print('control:', control)
print("length of MSE", len(MSEs))
print("avg MSE: ", np.mean(MSEs))
print("length of MSE special", len(MSE_specials))
print("avg MSE special: ", np.mean(MSE_specials))
