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

class ThreeLayer:
    '''
    Three layer baysian higherarchical model.
    Both levels of structure need to be indexed at 0
    eg States and Counties for Alaska: county 1, county 2,
    Alabama: county 1, county 2 should be
    |State|Counnty|
    |0|0|
    |0|1|
    |1|0|
    |1|1|

    Uses an intercept and three betas
    '''

    def __init__(self):
        self.sc_amean = []
        self.sc_b1mean = []
        self.sc_b2mean = []
        self.sc_b3mean = []

    def fit(self,X_train):
        unique_state = X_train.state_idx.unique()
        state_idx = X_train.state_idx.values
        county_idx = X_train.cty_idx.values
        unique_cty = X_train.cty_idx.unique()

        with pm.Model() as SC_hierarchical_model:
        # Hyperpriors

        mu_a = pm.Normal('mu_alpha', mu=0., sd=1)
        sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)

        mu_b1 = pm.Normal('mu_beta1', mu=0., sd=1)
        sigma_b1 = pm.HalfCauchy('sigma_beta1', beta=1)

        mu_b2 = pm.Normal('mu_beta2', mu=0., sd=1)
        sigma_b2 = pm.HalfCauchy('sigma_beta2', beta=1)

        mu_b3 = pm.Normal('mu_beta3', mu=0., sd=1)
        sigma_b3 = pm.HalfCauchy('sigma_beta3', beta=1)

        # Intercept for each state county, distributed around group mean mu_a
        sa = pm.Normal('st_alpha', mu=mu_a, sd=sigma_a, shape=[5,222])
        # Beta for each state county, distributed around group mean mu_beta_i
        sb1 = pm.Normal('st_beta1', mu=mu_b1, sd=sigma_b1, shape=[5,222])
        sb2 = pm.Normal('st_beta2', mu=mu_b2, sd=sigma_b2, shape=[5,222])
        sb3 = pm.Normal('st_beta3', mu=mu_b3, sd=sigma_b3, shape=[5,222])

        # Model error
        eps = pm.Uniform('sigma_eps', lower = 0, upper = 5000)

        # Expected value
        x1 = X_train.year.values
        x2 = X_train.MA_Part_Rate.values
        x3 = X_train.IP_per_1000.values
        cost_est = sa[state_idx,county_idx] + sb1[state_idx,county_idx]*x1 + sb2[state_idx,county_idx]*x2 + sb3[state_idx,county_idx]*x3

        # Data likelihood
        y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.Cost_per_Beneficiary)

        start = pm.find_MAP()
        step = pm.NUTS()
        with SC_hierarchical_model:
            SC_hierarchical_trace = pm.sample(1000,step,start,njobs=3)

        #get traces
        SCalphas = SC_hierarchical_trace.get_values('st_alpha', burn = 50)
        SCbeta1s = SC_hierarchical_trace.get_values('st_beta1', burn = 50)
        SCbeta2s = SC_hierarchical_trace.get_values('st_beta2', burn = 50)
        SCbeta3s = SC_hierarchical_trace.get_values('st_beta3', burn = 50)

        # get county parameters

        for i in range(len(state_idx)):
            self.sc_amean.append(np.mean(SCalphas[:,state_idx[i],county_idx[i]]))
            self.sc_b1mean.append(np.mean(SCbeta1s[:,state_idx[i],county_idx[i]]))
            self.sc_b2mean.append(np.mean(SCbeta2s[:,state_idx[i],county_idx[i]]))
            self.sc_b3mean.append(np.mean(SCbeta3s[:,state_idx[i],county_idx[i]]))

        return self


    def predict(self,X_test):
        #get predicted values
        y_hats_2H = self.sc_amean + self.sc_b1mean * X_test.year +
            self.sc_b2mean * X_test.MA_Part_Rate
            + self.sc_b3mean * X_test.IP_per_1000

        #get RMSE
        y_test['y_pred_2H'] = y_hats_2H
        RMSE = np.sqrt(mean_squared_error(y_test.y_pred_2H,y_test.Cost_per_Beneficiary))
        print(RMSE)

        return y_hats_2H


    def plot_predictions(self,X_test):
        y_hats_2H = self.predict(X_test)

        #plot results
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(y_test.Cost_per_Beneficiary,y_test.y_pred_2H,'ko')
        ax.set_xlabel('Actual 2014 Cost')
        ax.set_ylabel('Predicted 2014 Cost')
        ax.set_title('State and County Hierarchical Models')
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)

def reindex_multilayer(df):
    #reindex cty to start at 0 for each state
    pd.options.mode.chained_assignment = None
    for i in df.state_idx.unique():
        temp = df[df.state_idx == i]
        temp['cty_idx'] = temp.cty_idx - int(temp.cty_idx.min())
        df.loc[df.state_idx == i, 'cty_idx'] = temp['cty_idx']
