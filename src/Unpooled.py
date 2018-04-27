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

class UnpooledModel:
    def __init__(self):
        self.trace = None

    def fit(self, X_train):
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
            self.trace = indiv_traces



        def predict(self, X_test):
            y_hat_inds = []
            for key, trace in self.trace.items():
                a = np.mean(trace.get_values('alpha'))
                b1 = np.mean(trace.get_values('beta1'))
                b2 = np.mean(trace.get_values('beta2'))
                b3 = np.mean(trace.get_values('beta3'))

                y_hat_ind = a + b1*X_test.iloc[key, 2] + b2*X_test.iloc[key, 3] + b3*X_test.iloc[key, 4]
                y_hat_inds.append(y_hat_ind)

            return y_hat_inds



        def plot(self,y,y_pred):
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            ax.plot(y,y_pred,'ko')
            ax.set_xlabel('Actual 2014 Cost')
            ax.set_ylabel('Predicted 2014 Cost')
            ax.set_title('Individual Models')
            ax.set_xlim(-3,3)
            ax.set_ylim(-3,3)
