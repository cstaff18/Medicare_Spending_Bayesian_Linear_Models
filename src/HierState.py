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

state_idx = X_train.state_idx.values

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)

    mu_b1 = pm.Normal('mu_beta1', mu=0., sd=1)
    sigma_b1 = pm.HalfCauchy('sigma_beta1', beta=1)

    mu_b2 = pm.Normal('mu_beta2', mu=0., sd=1)
    sigma_b2 = pm.HalfCauchy('sigma_beta2', beta=1)

    mu_b3 = pm.Normal('mu_beta3', mu=0., sd=1)
    sigma_b3 = pm.HalfCauchy('sigma_beta3', beta=1)

    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(state_idx.unique()))
    # Beta for each county, distributed around group mean mu_beta_i
    b1 = pm.Normal('beta1', mu=mu_b1, sd=sigma_b1, shape=len(state_idx.unique()))
    b2 = pm.Normal('beta2', mu=mu_b2, sd=sigma_b2, shape=len(state_idx.unique()))
    b3 = pm.Normal('beta3', mu=mu_b3, sd=sigma_b3, shape=len(state_idx.unique()))

    # Model error
    eps = pm.Uniform('sigma_eps', lower = 0, upper = 5000)

    # Expected value
    x1 = X_train.year.values
    x2 = X_train.MA_Part_Rate.values
    x3 = X_train.IP_per_1000.values
    cost_est = a[state_idx] + b1[state_idx]*x1 + b2[state_idx]*x2 + b3[state_idx]*x3

    # Data likelihood
    y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.Cost_per_Beneficiary)
