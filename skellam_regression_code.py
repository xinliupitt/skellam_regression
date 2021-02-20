"""Implementation of Skellam regression training and prediction processes."""

import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Skellam regression training process

# z: ground truth matrix with size (number of training records, 1).
# The value is the difference between bike demand and dock demand.
z = pd.read_pickle("data/z_train.pkl")
# infos: independent variable matrix with size (number of training records, num_features).
infos = pd.read_pickle("data/infos_train.pkl")
# num_features: number of features.
num_features = len(infos[0])
# lambda_reg: regularization shrinking parameter.
lambda_reg = 0

def MLERegression(params):
    """calculate loss for Skellam regression training"""
    # weight matrix to be trained for bike demand
    beta1 = params[:num_features]
    # weight matrix to be trained for dock demand
    beta2 = params[num_features:2*num_features]
    # bias matrix to be trained for bike demand
    int1 = params[-2]
    # bias matrix to be trained for dock demand
    int2 = params[-1]
    # calculate bike demand
    l1 = int1 + np.dot(infos, beta1)
    # calculate dock demand
    l2 = int2 + np.dot(infos, beta2)
    # calculate the loss by Skellam probability mass function
    negLL = -np.sum(stats.skellam.logpmf(z, mu1=np.exp(l1), mu2=np.exp(l2), loc=0)) + lambda_reg*(norm(beta1, 1) + norm(beta2, 1))
    return(negLL)

# x0: initialized weight matrix
x0 = np.asarray([0.4] * (2*num_features+2))

# results: trained weight matrix
results = minimize(MLERegression, x0, method="CG",
                   options={'disp': True, 'maxiter': 20})

# Skellam regression prediction/testing process

# infos: independent variable matrix with size (number of testing records, num_features).
infos = pd.read_pickle("data/infos_test.pkl")

# z_test_truth: ground truth matrix with size (number of testing records, 1).
# The value is the difference between bike demand and dock demand.
z_test_truth = pd.read_pickle("data/z_test_truth.pkl")

# z_hat: predicted dependent variable matrix.
z_hat = []
for info in infos:
    # apply weight matrix "results" to calculate dependent variable
    mu1_hat = np.exp(results.x[-2] + np.dot(results.x[:num_features], info))
    mu2_hat = np.exp(results.x[-1] + np.dot(results.x[num_features:2*num_features], info))
    z_hat.append(mu1_hat-mu2_hat)

print("z prediction:", z_hat)
print("z ground truth:", z_test_truth)
