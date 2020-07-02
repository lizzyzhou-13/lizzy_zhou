---
title: "Efficient Frontier from Modern Portfolio Theory implemented in Python"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - finance
---


My personal interest in finance has led me to take the [Financial Risk Manager](https://www.garp.org/?gclid=Cj0KCQjw0Mb3BRCaARIsAPSNGpVyi-3aZIXl2iKpaFkfWWSoqvtMX_XXz6iQtz-HZToBUqXhEcdU8UsaAmRdEALw_wcB#!/frm)(FRM) exam and online courses provided on Coursera. During my study, I found that Efficient Frontier from Modern Portfolio Theory is the most basic knowledge for whoever touches down to the financial sector.

While I was going through my study, I thought it would be a very good
 reference to practice my Python skills. Even though the FRM exam or the 
 Coursera course did not provide any technical details of how to implement it 
 in Python, with some search around I found a couple of very useful blog posts 
 I can refer to.

Series of Medium blog post by [Bernard Brenyah](https://medium.com/@bbrenyah)
- [Markowitz’s Efficient Frontier in Python [Part 1/2]](https://medium.com/python-data/effient-frontier-in-python-34b0c3043314)
- [Markowitz’s Efficient Frontier in Python [Part 2/2]](https://medium.com/python-data/efficient-frontier-portfolio-optimization-with-python-part-2-2-2fe23413ad94)

Blog post by [Bradford Lynch](http://www.bradfordlynch.com/)
- [Investment Portfolio Optimization](http://www.bradfordlynch.com/blog/2015/12/04/InvestmentPortfolioOptimization.html)

Based on what I have learned, and also from the above blog posts, I have tried 
to replicate it in my own way, tweaking bit and pieces along the way.

## Modern Portfolio Theory

Modern Portfolio Theory (MPT) is an investment theory developed by Harry 
Markowitz and published under the title "Portfolio Selection" in the Journal 
of Finance in 1952.

There are a few underlying concepts that can help anyone to understand MPT. 
If you are familiar with finance, you might know what the acronym "TANSTAAFL" 
stands for. It is a famous acronym for "There Ain't No Such Thing As A Free 
Lunch". This concept is also closely related to 'risk-return trade-off'.

Higher risk is associated with greater probability of higher return and lower 
risk with a greater probability of smaller return. MPT assumes that investors 
re risk-averse, meaning that given two portfolios that offer the same expected 
return, investors will prefer the less risky one. Thus, an investor will take 
on increased risk only if compensated by higher expected returns. 

Another factor comes in to play in MPT is "diversification". Modern portfolio 
theory says that it is not enough to look at the expected risk and return of 
one particular stock. By investing in more than one stock, an investor can reap 
the benefits of diversification – chief among them, a reduction in the riskiness 
of the portfolio.

What needs to be understood is "risk of a portfolio is not equal to average
/weighted-average of individual stocks in the portfolio". In terms of return, 
yes it is the average/weighted average of individual stock's returns, 
but that's not the case for risk. The risk is about how volatile the asset is, 
if you have more than one stock in your portfolio, then you have to take count 
of how these stocks movement correlates with each other. The beauty of 
diversification is that you can even get lower risk than a stock with the 
lowest risk in your portfolio, by optimising the allocation. 

First, let's start by importing some libraries we need. "Quandl" is a financial 
platform which also offers Python library. Along side quandl, other important 
dependencies are imported.


```python
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco

plt.style.use('fivethirtyeight')
np.random.seed(777)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

The stocks selected for this post is Apple, Amazon, Google, Facebook. 
Below code block will get daily adjusted closing price of each stock 
from 01/01/2016 to 31/12/2017.


```python
quandl.ApiConfig.api_key = 'API_key'
stocks = ['AAPL','AMZN','GOOGL','FB']
data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, paginate=True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>ticker</th>
      <th>adj_close</th>
    </tr>
    <tr>
      <th>None</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-12-29</td>
      <td>GOOGL</td>
      <td>1053.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-12-28</td>
      <td>GOOGL</td>
      <td>1055.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-12-27</td>
      <td>GOOGL</td>
      <td>1060.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-12-26</td>
      <td>GOOGL</td>
      <td>1065.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-12-22</td>
      <td>GOOGL</td>
      <td>1068.86</td>
    </tr>
  </tbody>
</table>
</div>



By looking at the info() of data, it seems like the "date" column is already in 
datetime format.


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2006 entries, 0 to 2005
    Data columns (total 3 columns):
    date         2006 non-null datetime64[ns]
    ticker       2006 non-null object
    adj_close    2006 non-null float64
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 47.1+ KB



```python
df = data.set_index('date')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>adj_close</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-29</th>
      <td>GOOGL</td>
      <td>1053.40</td>
    </tr>
    <tr>
      <th>2017-12-28</th>
      <td>GOOGL</td>
      <td>1055.95</td>
    </tr>
    <tr>
      <th>2017-12-27</th>
      <td>GOOGL</td>
      <td>1060.20</td>
    </tr>
    <tr>
      <th>2017-12-26</th>
      <td>GOOGL</td>
      <td>1065.85</td>
    </tr>
    <tr>
      <th>2017-12-22</th>
      <td>GOOGL</td>
      <td>1068.86</td>
    </tr>
  </tbody>
</table>
</div>



By specifying col[1] in below list comprehension you can select the stock
 names under multi-level column
```python
table = df.pivot(columns='ticker')
table.columns = [col[1] for col in table.columns]
table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOGL</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>101.783763</td>
      <td>636.99</td>
      <td>102.22</td>
      <td>759.44</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>99.233131</td>
      <td>633.79</td>
      <td>102.73</td>
      <td>761.53</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>97.291172</td>
      <td>632.65</td>
      <td>102.97</td>
      <td>759.33</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>93.185040</td>
      <td>607.94</td>
      <td>97.92</td>
      <td>741.00</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>93.677776</td>
      <td>607.05</td>
      <td>97.33</td>
      <td>730.91</td>
    </tr>
  </tbody>
</table>
</div>



Let's first look at how the price of each stock has evolved within give time frame.


```python
plt.figure(figsize=(14, 7))
for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')
```


![png]({{site.url}}{{site.baseurl}}/assets/images/efficient_frontier/output_18_1.png)


It looks like that Amazon and Google's stock price is relatively more expensive 
than those of Facebook and Apple. But since Facebook and Apple are squashed 
at the bottom, it is hard to see the movement of these two.

Another way to plot this is plotting daily returns (percent change compared 
to the day before). By plotting daily returns instead of actual prices, we can 
see the stocks' volatility.


```python
returns = table.pct_change()

plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')
```

    Text(0, 0.5, 'daily returns')




![png]({{site.url}}{{site.baseurl}}/assets/images/efficient_frontier/output_21_2.png)


Amazon has two distinctive positive spikes and a couple of negative ones. 
Facebook has one highest positive spike. And Google seems to be the least volatile.

## Random Portfolios Generation

We have 4 stocks in our portfolio. One decision we have to make is how we 
should allocate our budget to each of stock in our portfolio. If our total 
budget is 1, then we can decide the weights for each stock, so that the sum 
of weights will be 1. And the value for weights will be the portion of budget 
we allocate to a specific stock. For example, if weight is 0.5 for Amazon, it 
means that we allocate 50% of our budget to Amazon.

Let's define some functions to simulate random weights to each stock in the 
portfolio, then calculate the portfolio's overall annualised returns and 
annualised volatility.

"portfolio_annual_performance" function will calculate the returns and 
volatility, and to make it as an annualised calculation I take into account 
252 as the number of trading days in one year. "random_portfolios" function 
will generate portfolios with random weights assigned to each stock, and by 
passing num_portfolios argument, you can decide how many random portfolios 
you want to generate.


```python
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
```


```python
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in xrange(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
```

You can easily get daily returns by calling pct_change on the data frame with 
the price data. And the mean daily returns, the covariance matrix of returns 
are needed to calculate portfolio returns and volatility. Finally, let's 
generate 25,000 portfolios with random weights assigned to each stock.


```python
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178
```

Let me briefly explain what below function is doing. First, it generates random 
portfolio and gets the results (portfolio returns, portfolio volatility, 
portfolio Sharpe ratio) and weights for the corresponding result. Then by 
locating the one with the highest Sharpe ratio portfolio, it displays maximum 
Sharpe ratio portfolio as red star sign. And does similar steps for minimum 
volatility portfolio, and displays it as the green star on the plot. All the 
randomly generated portfolios will be also plotted with colour map applied to 
them based on the Sharpe ratio. The bluer, the higher Sharpe ratio.

And for these two optimal portfolios, it will also show how it allocates the 
budget within the portfolio.


```python
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print "-"*80
    print "Maximum Sharpe Ratio Portfolio Allocation\n"
    print "Annualised Return:", round(rp,2)
    print "Annualised Volatility:", round(sdp,2)
    print "\n"
    print max_sharpe_allocation
    print "-"*80
    print "Minimum Volatility Portfolio Allocation\n"
    print "Annualised Return:", round(rp_min,2)
    print "Annualised Volatility:", round(sdp_min,2)
    print "\n"
    print min_vol_allocation
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
```



```python
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
```

    --------------------------------------------------------------------------------
    Maximum Sharpe Ratio Portfolio Allocation
    
    Annualised Return: 0.3
    Annualised Volatility: 0.18
    
    
                 AAPL   AMZN     FB  GOOGL
    allocation  43.93  29.49  26.51   0.07
    --------------------------------------------------------------------------------
    Minimum Volatility Portfolio Allocation
    
    Annualised Return: 0.22
    Annualised Volatility: 0.16
    
    
                 AAPL  AMZN    FB  GOOGL
    allocation  34.12  0.04  8.25  57.59



![png]({{site.url}}{{site.baseurl}}/assets/images/efficient_frontier/output_33_1.png)


For minimum risk portfolio, we can see that more than half of our budget is 
allocated to Google. If you take another look at the daily return plot from 
earlier, you can see that Google is the least volatile stock of four, so 
allocating a large percentage to Google for minimum risk portfolio makes 
intuitive sense.

If we are willing to take higher risk for higher return, one that gives us 
the best risk-adjusted return is the one with maximum Sharpe ratio. In this 
scenario, we are allocating a significant portion to Amazon and Facebook, which 
are quite volatile stocks from the previous plot of daily returns. And Google 
which had more than 50% allocation in the minimum risk portfolio, has less than 
1% budget allocated to it.

## Efficient Frontier

From the plot of the randomly simulated portfolio, we can see it forms a shape 
of an arch line on the top of clustered blue dots. This line is called efficient 
frontier. Why is it efficient? Because points along the line will give you the 
lowest risk for a given target return. All the other dots right to the line will 
give you higher risk with same returns. If the expected returns are the same, 
why would you take an extra risk when there's an option with lower risk?

The way we found the two kinds of optimal portfolio above was by simulating 
many possible random choices and pick the best ones (either minimum risk or 
maximum risk-adjusted return). We can also implement this by using Scipy's 
optimize function.

If you are an advanced Excel user, you might be familiar with 'solver' function 
in excel. Scipy's optimize function is doing the similar task when given what 
to optimize, and what are constraints and bounds.

Below functions are to get the maximum Sharpe ratio portfolio. In Scipy's 
optimize function, there's no 'maximize', so as an objective function you need 
to pass something that should be minimized. That is why the first 
"neg_sharpe_ratio" is computing the negative Sharpe ratio. Now we can use this 
as our objective function to minimize. In "max_sharpe_ratio" function, you first 
define arguments (this should not include the variables you would like to change 
for optimisation, in this case, "weights"). At first, the construction of 
constraints was a bit difficult for me to understand, due to the way it is stated. 

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

The above constraint is saying that sum of x should be equal to 1. You can think 
of the 'fun' part construction as '1' on the right side of equal sign has been 
moved to the left side of the equal sign.

'np.sum(x) == 1' has become 'np.sum(x)-1'

And what does this mean? It simply means that the sum of all the weights should 
be equal to 1. You cannot allocate more than 100% of your budget in total.

"bounds" is giving another limit to assign random weights, by saying any weight 
should be inclusively between 0 and 1. You cannot give minus budget allocation 
to a stock or more than 100% allocation to a stock.


```python
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result
```

We can also define the optimizing function for calculating minimum volatility 
portfolio. This time we really do minimize the objective function. What do we 
want to minimize? We want to minimize volatility by trying different weights. 
"constraints" and "bounds" are same as the above.


```python
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result
```

As I already mentioned above we can also draw a line which depicts where the 
efficient portfolios for a given risk rate should be. This is called "efficient 
frontier". Below I define other functions to compute efficient frontier. 
The first function "efficient_return" is calculating the most efficient 
portfolio for a given target return, and the second function "efficient_frontier" 
will take a range of target returns and compute efficient portfolio for each 
return level.


```python
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients
```

Let's try to plot the portfolio choice with maximum Sharpe ratio and minimum 
volatility also with all the randomly generated portfolios. But this time we 
are not picking the optimal ones from the randomly generated portfolios, but we 
are actually calculating by using Scipy's 'minimize' function. And the below 
function will also plot the efficient frontier line.


```python
def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print "-"*80
    print "Maximum Sharpe Ratio Portfolio Allocation\n"
    print "Annualised Return:", round(rp,2)
    print "Annualised Volatility:", round(sdp,2)
    print "\n"
    print max_sharpe_allocation
    print "-"*80
    print "Minimum Volatility Portfolio Allocation\n"
    print "Annualised Return:", round(rp_min,2)
    print "Annualised Volatility:", round(sdp_min,2)
    print "\n"
    print min_vol_allocation
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
```



```python
display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
```


We have almost the same result as what we have simulated by picking from the 
randomly generated portfolios. The slight difference is that the Scipy's "optimize" 
function has not allocated any budget at all for Google on maximum Sharpe ratio 
portfolio, while one we chose from the randomly generated samples has 0.45% of 
allocation for Google. There are some differences in the decimal places but more 
or less same.

Instead of plotting every randomly generated portfolio, we can plot each 
individual stocks on the plot with the corresponding values of each stock's 
annual return and annual risk. This way we can see and compare how diversification 
is lowering the risk by optimising the allocation.


```python
def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print "-"*80
    print "Maximum Sharpe Ratio Portfolio Allocation\n"
    print "Annualised Return:", round(rp,2)
    print "Annualised Volatility:", round(sdp,2)
    print "\n"
    print max_sharpe_allocation
    print "-"*80
    print "Minimum Volatility Portfolio Allocation\n"
    print "Annualised Return:", round(rp_min,2)
    print "Annualised Volatility:", round(sdp_min,2)
    print "\n"
    print min_vol_allocation
    print "-"*80
    print "Individual Stock Returns and Volatility\n"
    for i, txt in enumerate(table.columns):
        print txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2)
    print "-"*80
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)
```


```python
display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate)
```

    --------------------------------------------------------------------------------
    Maximum Sharpe Ratio Portfolio Allocation
    
    Annualised Return: 0.3
    Annualised Volatility: 0.18
    
    
                 AAPL   AMZN     FB  GOOGL
    allocation  44.72  29.13  26.15    0.0
    --------------------------------------------------------------------------------
    Minimum Volatility Portfolio Allocation
    
    Annualised Return: 0.22
    Annualised Volatility: 0.16
    
    
                 AAPL  AMZN   FB  GOOGL
    allocation  33.94  0.66  7.0   58.4
    --------------------------------------------------------------------------------
    Individual Stock Returns and Volatility
    
    AAPL : annuaised return 0.28 , annualised volatility: 0.21
    AMZN : annuaised return 0.34 , annualised volatility: 0.25
    FB : annuaised return 0.3 , annualised volatility: 0.23
    GOOGL : annuaised return 0.18 , annualised volatility: 0.18
    --------------------------------------------------------------------------------



![png]({{site.url}}{{site.baseurl}}/assets/images/efficient_frontier/output_51_1.png)


As you can see from the above plot, the stock with the least risk is Google at 
around 0.18. But with portfolio optimisation, we can achieve even lower risk at 
0.16, and still with a higher return than Google. And if we are willing to take 
slightly more risk at around the similar level of risk of Google, we can achieve 
a much higher return of 0.30 with portfolio optimization.
