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


county_names = X_train.cty_idx.unique.values

indiv_traces = {}
for county_name in county_names:
    # Select subset of data belonging to county
    c_data = X_train[X_train.cty_idx == county_name]
    c_data = c_data.reset_index(drop=True)

    c_log_radon = c_data.log_radon
    c_floor_measure = c_data.floor.values

    x1 = c_data.year.values
    x2 = c_data.MA_Part_Rate.values
    x3 = c_data.IP_per_1000.values

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
        y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=c_data.Cost_per_Beneficiary)

        # Inference button (TM)!
        trace = pm.sample(progressbar=False)

    indiv_traces[county_name] = trace
