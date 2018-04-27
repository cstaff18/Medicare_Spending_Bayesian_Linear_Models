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



class PooledModel:
    def __init__(self):
        self.trace = None

    def fit(self, X_train):
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

            self.trace = trace


    def predict(self, X_test):

        a = np.mean(self.trace.get_values('alpha',burn = 50))
        b1 = np.mean(self.trace.get_values('beta1',burn = 50))
        b2 = np.mean(self.trace.get_values('beta2',burn = 50))
        b3 = np.mean(self.trace.get_values('beta3',burn = 50))

        x1 = X_test.year.values
        x2 = X_test.MA_Part_Rate.values
        x3 = X_test.IP_per_1000.values
        return a + b1 * x1 + b2 * x2 +b2 * x3



    def plot(self,y,y_pred):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(y,y_pred,'ko')
        ax.set_xlabel('Actual 2014 Cost')
        ax.set_ylabel('Predicted 2014 Cost')
        ax.set_title('Pooled Model')
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
