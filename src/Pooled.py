import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import pymc3 as pm
import theano
from sklearn.metrics import mean_squared_error


x1 = X_train.year.values
x2 = X_train.MA_Part_Rate.values
x3 = X_train.IP_per_1000.values

with pm.Model() as individual_model:
    # Intercept
    a = pm.Normal('alpha', mu=0, sd = 1)
    # Betas
    b1 = pm.Normal('beta1', mu=0, sd=1)
    b2 = pm.Normal('beta2', mu=0, sd = 1)
    b3 = pm.Normal('beta3', mu=0, sd = 1)

    # Model error prior
    eps = pm.HalfCauchy('eps', beta=1)

    # Linear model
    cost_est = a + b1*x1 + b2*x2 + b3*x3

    # Data likelihood
    y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.Cost_per_Beneficiary)

    # Inference button (TM)!
    trace = pm.sample(progressbar=False)
