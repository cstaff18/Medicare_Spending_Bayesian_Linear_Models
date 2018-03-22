import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import pymc3 as pm
from sklearn.metrics import mean_squared_error


def load_data():
     #load cleaned data set
     df = pd.read_csv('data/medicare_county_level/medicare_county_all.csv',
        index_col=0)
     #Convert fields to numeric type
     df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric, errors='coerce', axis=1)
     #Create Target Vairable Cost/Beneficiary
     df['Cost_per_Beneficiary'] = df['Total Actual Costs'] /
        ['Beneficiaries with Part A and Part B']
     return df


def feature_reduction(df):
    #get identifying features
    state = df['State']
    county = df['County']
    year = df['year']
    countyid = df['State and County FIPS Code']

    #drop columns that are greater than 20% Nan
    nullrate = df.isnull().sum()/df.shape[0]
    mask = nullrate < 0.2
    cols_tokeep_null = nullrate[mask]
    df = df.filter(list(cols_tokeep_null.index))

    #Select Correlated columns
    #Target must be last row
    correlation_matrix = df.corr()
    correlation_target = correlation_matrix.iloc[:,-1]

    #Keep anything with greater than 0.4 correlation coef
    mask = (correlation_target > 0.3) | (correlation_target < -0.3)
    fields_to_keep = correlation_target[mask]
    colnames = fields_to_keep.index
    df = df.filter(list(colnames))

    #Add identifying features back in
    df['State'] = state
    df['County'] = county
    df['year'] = year
    df['CountyID'] = countyid
    #cast year as int as type
    df.year = df.year.astype(int)
    return df


def make_violin_plots(df):
    #order states by most expensive
    order = df.groupby('State').mean()
        ['Cost_per_Beneficiary'].sort_values(ascending=False)
    order.drop(['National','XX','PR'],inplace = True)

    matplotlib.rcParams['figure.figsize'] = [25.0, 8.0]

    #Create State Wide Plot
    sns.violinplot(x="State", y="Cost_per_Beneficiary",
        data=df_med_county, order= list(order.index), width = 1.9,
        palette="coolwarm", cut=0)
    plt.title('State Wide Distribution of Cost/Beneficiary', fontsize=40)
    plt.ylabel('Avg Cost/Beneficiary per County')
    plt.show()

    #Create Nation Wide plot
    sns.violinplot(order, orient='v')
    plt.title('National Distribution of Cost/Beneficiary', fontsize=40)
    plt.ylabel('Avg Cost/Beneficiary per State')
    plt.xlabel('USA')
    plt.show()


def index_states_counties(df):
    #Select features for use in Bayes Linear Model
    df = df.filter(['Cost_per_Beneficiary','CountyID','County','State','year',
        'MA Participation Rate','IP Covered Stays Per 1000 Beneficiaries'])
    #Standardize data
    df.year = preprocessing.scale(df.year)
    df['MA_Part_Rate'] = preprocessing.scale(df['MA Participation Rate'])
    df['IP_per_1000'] = preprocessing.scale(
        df['IP Covered Stays Per 1000 Beneficiaries'])

    df.drop(['MA Participation Rate', 'IP Covered Stays Per 1000 Beneficiaries']
        ,axis = 1, inplace = True)

    # Create a lookup table with the index 0 to n_states matching to unique states
    states = df['State'].unique()
    n_states = len(states)

    state_lookup = pd.DataFrame({'state_idx': range(n_states), 'state': states})
    df = df.merge(state_lookup, how = 'left', left_on = 'State',
        right_on = 'state')
    df = df.drop(['State', 'state'], axis = 1)

    #Select only the first 5 states
    df = df[df.state_idx < 5]

    # Create a lookup table with the index 0 to n_counties matching to unique fips code
    counties = df['CountyID'].unique()
    n_counties = len(counties)

    cty_lookup = pd.DataFrame({'cty_idx': range(n_counties), 'fips': counties})
    # Merge the original dataframe with the cty_lookup table

    df = df.merge(cty_lookup, how = 'left', left_on = 'CountyID', right_on = 'fips')
    # Drop the fips code
    df = df.drop(['CountyID', 'fips'], axis = 1)

    return df


def split_data(df):
    #Standardize Y for use in bayes model so no assumptions are made for distributions
    df['Cost_per_Beneficiary'] = preprocessing.scale(df['Cost_per_Beneficiary'])

    X = df
    year = df.year.unique()

    #Split data with the last year as the testing data
    y = X.filter(['cty_idx','state_idx','Cost_per_Beneficiary'])
    X_train = X[X.year != max(year)]
    X_test = X[X.year == max(year)]
    y_train = y[X.year != max(year)]
    y_test = y[X.year == max(year)]

    return X_train, X_test, y_train, y_test

def cty_hier_model(X_train, X_test, y_train, y_test):

    county_idx = X_train.cty_idx.values
    unique_cty = X_train.cty_idx.unique()

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
        a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(unique_cty))
        # Beta for each county, distributed around group mean mu_beta_i
        b1 = pm.Normal('beta1', mu=mu_b1, sd=sigma_b1, shape=len(unique_cty))
        b2 = pm.Normal('beta2', mu=mu_b2, sd=sigma_b2, shape=len(unique_cty))
        b3 = pm.Normal('beta3', mu=mu_b3, sd=sigma_b3, shape=len(unique_cty))

        # Model error
        eps = pm.Uniform('sigma_eps', lower = 0, upper = 5000)

        # Expected value
        x1 = X_train.year.values
        x2 = X_train.MA_Part_Rate.values
        x3 = X_train.IP_per_1000.values
        cost_est = a[county_idx] + b1[county_idx]*x1 + b2[county_idx]*x2
            + b3[county_idx]*x3

        # Data likelihood
        y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.Cost_per_Beneficiary)

        start = pm.find_MAP()
        step = pm.NUTS()

    with hierarchical_model:
        hierarchical_trace = pm.sample(1000,step,start,njobs=3)

    #get parameters
    alphas = hierarchical_trace.get_values('alpha', burn = 50)
    beta1s = hierarchical_trace.get_values('beta1', burn = 50)
    beta2s = hierarchical_trace.get_values('beta2', burn = 50)
    beta3s = hierarchical_trace.get_values('beta3', burn = 50)

    #get parameter means
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

    #get predicted values
    x1 = X_test.year.values
    x2 = X_test.MA_Part_Rate.values
    x3 = X_test.IP_per_1000.values
    y_hats = alpha_means + beta1_means * x1 + beta2_means * x2 +
        beta3_means * x3

    #Get RMSE score
    RMSE = np.sqrt(mean_squared_error(y_test.y_pred_H1,y_test.Cost_per_Beneficiary))
    print('RMSE:', RMSE)

    #Plot predicted vs actual
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(y_test.Cost_per_Beneficiary,y_test.y_pred_H1,'ko')
    ax.set_xlabel('Actual 2014 Cost')
    ax.set_ylabel('Predicted 2014 Cost')
    ax.set_title('Hierarchical County Model')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    plt.show()

    #add results to y_test df
    y_test['y_pred_H1'] = y_hats

    return y_test


def pooled_model(X_train, X_test, y_train, y_test):
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

        start = pm.find_MAP()
        step = pm.NUTS()

    with individual_model:
        individual_trace = pm.sample(500,step,start,njobs=3)

    #get parameters
    a = np.mean(individual_trace.get_values('alpha',burn = 50))
    b1 = np.mean(individual_trace.get_values('beta1',burn = 50))
    b2 = np.mean(individual_trace.get_values('beta2',burn = 50))
    b3 = np.mean(individual_trace.get_values('beta3',burn = 50))

    #get predictions
    y_hats_pool = a + b1 * x1 + b2 * x2 +b2 * x3

    #add predictions to y_test df
    y_test['y_pred_pool'] = y_hats_pool

    #print RMSE score
    RMSE = np.sqrt(mean_squared_error(y_test.y_pred_pool,y_test.Cost_per_Beneficiary))
    print('RMSE:',RMSE)

    #plot prec vs actual
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(y_test.Cost_per_Beneficiary,y_test.y_pred_pool,'ko')
    ax.set_xlabel('Actual 2014 Cost')
    ax.set_ylabel('Predicted 2014 Cost')
    ax.set_title('Pooled Model')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)

    return y_test


def state_hier_model(X_train, X_test, y_train, y_test):
    unique_state = X_train.state_idx.unique()
    state_idx = X_train.state_idx.values

    with pm.Model() as S_hierarchical_model:
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
        a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(unique_state))
        # Beta for each county, distributed around group mean mu_beta_i
        b1 = pm.Normal('beta1', mu=mu_b1, sd=sigma_b1, shape=len(unique_state))
        b2 = pm.Normal('beta2', mu=mu_b2, sd=sigma_b2, shape=len(unique_state))
        b3 = pm.Normal('beta3', mu=mu_b3, sd=sigma_b3, shape=len(unique_state))

        # Model error
        eps = pm.Uniform('sigma_eps', lower = 0, upper = 5000)

        # Expected value
        x1 = X_train.year.values
        x2 = X_train.MA_Part_Rate.values
        x3 = X_train.IP_per_1000.values
        cost_est = a[state_idx] + b1[state_idx]*x1 + b2[state_idx]*x2 +
            b3[state_idx]*x3

        # Data likelihood
        y_like = pm.Normal('y_like', mu=cost_est, sd=eps, observed=X_train.
            Cost_per_Beneficiary)

        start = pm.find_MAP()
        step = pm.NUTS()

    with S_hierarchical_model:
        S_hierarchical_trace = pm.sample(1000,step,start,njobs=3)

    Salphas = S_hierarchical_trace.get_values('alpha', burn = 50)
    Sbeta1s = S_hierarchical_trace.get_values('beta1', burn = 50)
    Sbeta2s = S_hierarchical_trace.get_values('beta2', burn = 50)
    Sbeta3s = S_hierarchical_trace.get_values('beta3', burn = 50)

    Salpha_means = []
    for j in range(Salphas.shape[1]):
        Salpha_means.append(np.mean(Salphas[:,j]))

    Sbeta1_means = []
    for j in range(Sbeta1s.shape[1]):
        Sbeta1_means.append(np.mean(Sbeta1s[:,j]))

    Sbeta2_means = []
    for j in range(Sbeta2s.shape[1]):
        Sbeta2_means.append(np.mean(Sbeta2s[:,j]))

    Sbeta3_means = []
    for j in range(Sbeta3s.shape[1]):
        Sbeta3_means.append(np.mean(Sbeta3s[:,j]))

    y_hats_S_Hier = []
    for i in unique_state:
        s = X_test[X_test.state_idx == i]
        x1 = s.year.values
        x2 = s.MA_Part_Rate.values
        x3 = s.IP_per_1000.values
        y_hat_s = Salpha_means[i] + Sbeta1_means[i] * x1 + Sbeta2_means[i] * x2
            + Sbeta3_means[i] * x3
        y_hats_S_Hier.extend(y_hat_s)

    y_test['y_pred_SHier'] = y_hats_S_Hier

    RMSE = np.sqrt(mean_squared_error(y_test.y_pred_SHier,y_test.Cost_per_Beneficiary))
    print(RMSE)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(y_test.Cost_per_Beneficiary,y_test.y_pred_SHier,'ko')
    ax.set_xlabel('Actual 2014 Cost')
    ax.set_ylabel('Predicted 2014 Cost')
    ax.set_title('Hierarchical State Models')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)

    return y_train


def unpooled_state_model(X_train, X_test, y_train, y_test):
    state_names = X_train.state_idx.unique()
    state_indiv_traces = {}

    for state_name in state_names:
        # Select subset of data belonging to county
        c_data = X_train[X_train.cty_idx == state_name]
        c_data = c_data.reset_index(drop=True)

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


            start = pm.find_MAP()
            step = pm.NUTS()

        state_indiv_traces[state_name] = trace

        y_hat_state_inds = []
        for key, trace in state_indiv_traces.items():
            a = np.mean(trace.get_values('alpha'))
            b1 = np.mean(trace.get_values('beta1'))
            b2 = np.mean(trace.get_values('beta2'))
            b3 = np.mean(trace.get_values('beta3'))

            s = X_test[X_test.state_idx == key]
            x1 = s.year.values
            x2 = s.MA_Part_Rate.values
            x3 = s.IP_per_1000.values
            y_hat_s = a + b1 * x1 + b2 * x2 + b3 * x3
            y_hat_state_inds.extend(y_hat_s)

        y_test['y_pred_S_ind'] = y_hat_state_inds
        RMSE = np.sqrt(mean_squared_error(y_test.y_pred_S_ind,y_test.Cost_per_Beneficiary))
        print(RMSE)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(y_test.Cost_per_Beneficiary,y_test.y_pred_S_ind,'ko')
        ax.set_xlabel('Actual 2014 Cost')
        ax.set_ylabel('Predicted 2014 Cost')
        ax.set_title('Individual State Models')
        ax.set_xlim(-3,3)
        return y_test

def plot_residuals(y_test)
    y_test['pool_Res'] = y_test.y_pred_pool - y_test.Cost_per_Beneficiary
    y_test['ind_cty_Res'] = y_test.y_pred_indv_cnty - y_test.Cost_per_Beneficiary
    y_test['ind_state_Res'] = y_test.y_pred_S_ind - y_test.Cost_per_Beneficiary
    y_test['hier_cty_Res'] = y_test.y_pred_H1 - y_test.Cost_per_Beneficiary
    y_test['hier_st_Res'] = y_test.y_pred_SHier - y_test.Cost_per_Beneficiary

    #figure 1
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    #ax.hist(y_test.pool_Res,alpha = 0.5)
    ax.hist(y_test.ind_cty_Res, bins = 15,label = 'independent models',alpha = 0.5)
    ax.hist(y_test.hier_cty_Res, bins = 15,label = 'hierarchical model',alpha = 0.5)

    ax.set_xlabel('Pred - Actual')
    #ax.set_ylabel()
    ax.set_title('County Residuals')
    ax.set_xlim(-3,3)
    ax.legend()

    #figure 2
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    #ax.hist(y_test.pool_Res,alpha = 0.5)
    ax.hist(y_test.ind_state_Res,bins = 15, label = 'independent models',alpha = 0.5)
    ax.hist(y_test.hier_st_Res,bins = 15, label = 'hierarchical model',alpha = 0.5)

    ax.set_xlabel('Pred - Actual')
    #ax.set_ylabel()
    ax.set_title('State Residuals')
    ax.legend()
    ax.set_xlim(-3,3)



if __name__ == '__main__':
    df_med_county = load_data()
    df_med_county = feature_reduction(df_med_county)
    make_violin_plots(df_med_county)

    df = index_states_counties(df_med_county)

    X_train, X_test, y_train, y_test = split_data(df)
