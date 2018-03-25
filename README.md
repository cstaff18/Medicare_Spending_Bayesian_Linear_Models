# Medicare Cost Modeling
## Adapted from Brendan Drew's [Capstone](https://github.com/brendan-drew/County-Medicare-Spending)

## Data
Medicare spending data was collected at county level from 2007 until 2014.  Some current data is available from Medicare's public dataset however additional data from Dartmouth's Medicare study was only collected until 2014.  The data contains the number of beneficiaries, spending, medical data, and population demographics for US counties.

## Question
Brendan was able to construct a hierarchical linear model to estimate 2014 spending data.  He used hyper parameters from the distributions of all counties and used that information to influence his county level regressions.



![State Wide Distribution](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/SWdist.png)

You can see that counties vary within a state.

![Nation Wide Distribution](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/NWdist.png)

Cost/Beneficiary also vary across states as shown in the National Distribution plot.  My goal is to adapt Brendan's two tier linear model to a three tier model of Medicare spending.


## Modeling
Use bayesian linear regression models trained on 2007 - 2013 data for 5 states with 3 features: year, MA participation, and IP per 1000 beneficiaries.


#### Pooled Model
For the first model we will create a baseline pooled model.  This creates 1 linear model and assumes there are no differences between states or individual counties.

![pooled model](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/poolgraph.png)

***pool RMSE = 1.233 or 1452.11$/Beneficiary***


#### Individual County Models
Next we well create independent, individual models for each county.  This means one linear model for each of the ~200 counties included.

![Individual County Models](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/IndCtygraph.png)

***Individual County RMSE = 0.601 or 708.91$/Beneficiary***

#### Hierarchical County Models
This approach creates a linear model for each county, however these individual models are no longer independent from each other.  Their individual parameters are influenced by the national distribution of parameters.

![Hierarchical County Models](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/H1graph.png)

***Hierarchical County RMSE= 0.641 or 755.66$/Beneficiary***

You can see that this model doesn't actually improve the RMSE score relative to the individual models but it does distribute the residuals more evenly. This could potentially mean that this approach would generalize better to more unseen data.

#### Individual State Models
Now on to the state models.  There appears to be structure in the state level medicare spending as well as shown by the Nation Wide Distribution of Spending Plot earlier.
Here I created independent, individual models for each of the five states included and plotted their counties estimated 2014 cost/Beneficiary vs actual 2014 cost/Beneficiary.

![Individual State Models](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/IndStategraph.png)

***Individual State RMSE = 0.601 or 708.91$/Beneficiary***

#### Hierarchical State Models
Finally we have the 5 individual state models whose parameters are influenced by the national distribution of parameters.
![Hierarchical State Models](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/H2graph.png)

***Hierarchical State RMSE = 0.421 or 495.80$/Beneficiary***

We can see that again the distribution of errors looks a bit more even especially at the upper range.  This hierarchical model was able to improve the RMSE score relative to the independent models.




## Did the Hierarchical Model improve predictions?
The hierarchical model for both county level regressions and state level regressions moved the distribution of residuals closer to zero.  The hierarchical model for state level regressions also narrowed the distribution and improved the RMSE score.  Hierarchical modeling of county level regressions did not improve the RMSE score. Below show the distribution of residuals for the independent and hierarchical models.

![County Residuals](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/countyresid.png)

![State Residuals](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/stateresid.png)

## Combining models
Now we can create a three level hierarchical model that included information from the national, state and county level using that same bayesian framework.  Let's see how this model performs.

National -> State -> County
![State and County Hierarchical Model](https://github.com/cstaff18/Medicare_Spending_Bayesian_Linear_Models/raw/master/images/SCHgraph.png)

***Hierarchical State and County RMSE= 0.185 or 217.87$/Beneficiary***

This model takes into the structure at the national, state, and county level and out performs all of our other models by a fair amount.
