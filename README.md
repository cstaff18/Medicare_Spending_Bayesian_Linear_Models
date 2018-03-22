# Medicare Cost Modeling
## Adapted from Brendan Drew's [Capstone](https://github.com/brendan-drew/County-Medicare-Spending)

## Data
Medicare spending data was collected at county level from 2007 until 2014.  Some current data is available from Medicare's public dataset however additional data from Dartmouth's Medicare study was only collected until 2014.  The data contains the number of beneficiaries, spending, medical data, and population demographics for US counties.

## Question
Brendan was able to construct a hierarchical linear model to estimate 2014 spending data.  He used hyper parameters from the distributions of all counties and used that information to influence his county level regressions.



![State Wide Distribution](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/SWdist.png)

You can see that counties vary within a state.

![Nation Wide Distribution](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/NWdist.png)

Cost/Beneficiary also vary across states as shown in the National Distribution plot.  My goal is to adapt Brendan's two tier linear model to a three tier model of Medicare spending.


## Modeling
Use bayesian linear regression models trained on 2007 - 2013 data for 5 states with 3 features: year, MA participation, and IP per 1000 beneficiaries.

![pooled model](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/poolgraph.png)

pool RMSE = 1.233 or 1452.11$/Beneficiary

![Individual County Models](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/IndCtygraph.png)

Individual County RMSE = 0.601 or 708.91$/Beneficiary

![Hierarchical County Models](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/H1graph.png)

Hierarchical County RMSE= 0.641 or 755.66$/Beneficiary


![Individual State Models](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/IndStategraph.png)

Individual State RMSE = 0.601 or 708.91$/Beneficiary

![Hierarchical State Models](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/H2graph.png)

Hierarchical State RMSE = 0.421 or 495.80$/Beneficiary

## Did the Hierarchical Model improve predictions?
The hierarchical model for both county level regressions and state level regressions moved the distribution of residuals closer to zero.  The hierarchical model for state level regressions also narrowed the distribution and improved the RMSE score.  Hierarchical modeling of county level regressions did not improve the RMSE score

![County Residuals](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/countyresid.png)

![State Residuals](https://github.com/cstaff18/County-Medicare-Spending/raw/master/images/stateresid.png)
