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

class HierCountyModel:
    def __init__(self):
        self.trace = None

    def fit(self, X_train):
        #Xtrain must have cty_idx column indexed at 0
        county_idx = X_train.cty_idx.values

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
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(county_idx))
            # Beta for each county, distributed around group mean mu_beta_i
            b1 = pm.Normal('beta1', mu=mu_b1, sd=sigma_b1, shape=len(county_idx))
            b2 = pm.Normal('beta2', mu=mu_b2, sd=sigma_b2, shape=len(county_idx))
            b3 = pm.Normal('beta3', mu=mu_b3, sd=sigma_b3, shape=len(county_idx))

            # Model error
            eps = pm.Uniform('sigma_eps', lower = 0, upper = 5000)

            # Expected value
            x1 = X_train.year.values
            x2 = X_train.MA_Part_Rate.values
            x3 = X_train.IP_per_1000.values
            cost_est = a[county_idx] + b1[county_idx]*x1 + b2[county_idx]*x2 + b3[county_idx]*x3

            # Data likelihood
            y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.Cost_per_Beneficiary)

            start = pm.find_MAP()
            step = pm.NUTS()
        with hierarchical_model:
            hierarchical_trace = pm.sample(1000,step,start,njobs=3)

        self.trace = hierarchical_trace



    def predict(self, X_test):
        
        alphas = self.trace.get_values('alpha', burn = 50)
        beta1s = self.trace.get_values('beta1', burn = 50)
        beta2s = self.trace.get_values('beta2', burn = 50)
        beta3s = self.trace.get_values('beta3', burn = 50)


        alpha_means = []
        for j in range(alphas.shape[1]):
            alpha_means.append(np.mean(alphas[:,j]))

        beta1_means = []
        for j in range(beta1s.shape[1]):
            beta1_means.append(np.mean(beta1s[:,j]))

        beta2_means = []
        for j in range(beta2s.shape[1]):
            beta2_means.append(np.mean(beta2s[:,j]))

        beta3_means = []
        for j in range(beta3s.shape[1]):
            beta3_means.append(np.mean(beta3s[:,j]))

        x1 = X_test.year.values
        x2 = X_test.MA_Part_Rate.values
        x3 = X_test.IP_per_1000.values
        y_hats = alpha_means + beta1_means * x1 + beta2_means * x2 +beta3_means * x3
        return y_hats

    def plot(self,y,y_pred):

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(y,y_pred,'ko')
        ax.set_xlabel('Actual 2014 Cost')
        ax.set_ylabel('Predicted 2014 Cost')
        ax.set_title('Hierarchical County Model')
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
