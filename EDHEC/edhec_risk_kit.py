import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize


def drawdown(return_series: pd.Series):
    """    
    Takes a time series of asset returns
    Computes and returns a dataframe that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    """
    
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    result = pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        "Drawdown":drawdowns
                          })
    
    
    return result


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns 
    of the top and bottom deciles by market cap
    """
    
    
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",header=0,index_col=0,parse_dates=True,na_values=-99.99)

    rets = me_m[['Lo 10','Hi 10']]
    rets.columns = ['SmallCap','LargeCap']
    rets.index = pd.to_datetime(rets.index, format = '%Y%m')
    rets = rets/100
    return rets



def get_hfi_returns():
    """
    Load the Fama-French Dataset for the returns 
    of the top and bottom deciles by market cap
    """
    
    
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",header=0,index_col=0,parse_dates=True)
    hfi.index = hfi.index.to_period('M')
    
    hfi /= 100

    return hfi

def get_ind_returns():
    """
    Load Ind 30 Returns
    """
    
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",header=0,index_col=0,parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%M").to_period("M")
    ind.columns = ind.columns.str.strip()
    
    return ind


def semideviation(r):
    """
    Returns the negative semideviation of r
    r must be a series or df
    """
    is_negative = r < 0
    
    return r[is_negative].std(ddof=0)


def skewness(r):
    """
    Alternate to scipy.stats.skew()
    """
    
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternate to scipy.stats.kurtosis()
    """
    
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    
    return exp/sigma_r**4


def is_normal(r, level = 0.01):
    """
    Applies Jarque-Bera test to determine normality
    Default 1% level
    """
    statistic, p_value = ss.jarque_bera(r)
    
    return p_value > level


def var_historic(r, level=5):
    """
    VaR Historic
    """

    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)

    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
def var_gaussian(r, level = 5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    z = ss.norm.ppf(level/100)

    if modified:
        s = skewness(r)
        k = kurtosis(r)
        
        z += (z**2 - 1)*s/6 + (z**3 -3*z)*(k-3)/24 - (2*z**3 -5*z)*(s**2)/36
    
    
    
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level = 5):
    """
    Computes the Conditional VaR of a Series or DataFrame
    """
    
    if isinstance(r,pd.Series):
        is_beyond = r <= -var_historic(r,level=level)
        return -r[is_beyond].mean()
    elif isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    annual_rets = ((1+r).prod())**(periods_per_year/len(r)) - 1

    return annual_rets


def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    """
    
    return r.std()*periods_per_year**0.5


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Annualized sharpe ratio for a set of returns
    """
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_excess_return = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    
    return ann_excess_return/ann_vol


def portfolio_returns(weights,returns):
    """
    Weights -> Returns
    """
    return weights.T @returns


def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, linestyle = ".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2:
        raise ValueError("Plot_ef2 can only plot 2-asset frontiers")

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]

    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    # pack into df
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})

    # plot
    return ef.plot.line(x="Volatility", y="Returns", style = linestyle)


def minimize_vol(target_return, er, cov):
    """
    Target Return -> Weight Vector
    """
    n = len(er)

    init_guess = np.repeat(1/n, n)

    # constrain to no leverage, and no short positions
    bounds = ((0, 1),)*n  # multiplying a tuple makes n copies of it

    # constraint that weights need to equal 1
    weights_sum_to_1 = {
        'type': 'eq',
        # check (sum of weights) - 1 equals 0
        'fun': lambda weights: np.sum(weights) - 1
    }

    # constraint that return should equal our target return
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        # check target return
        'fun': lambda weights, er: target_return - portfolio_returns(weights, er)
    }

    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method="SLSQP",
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)  # can pass arg: options = {'disp':False}
    return results.x


def optimal_weights(n_points, er, cov):
    """
    list of weights to run the optimizer on to minimze the vol
    """
    target_returns = np.linspace(er.min(),er.max(),n_points)
    
    weights = [minimize_vol(target_ret, er, cov) for target_ret in target_returns]
    
    return weights



def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that maximizes the Sharpe Ratio
    Given the riskfree rate, expected returns, and covariance matrix
    """
    n = len(er)

    init_guess = np.repeat(1/n, n)

    # constrain to no leverage, and no short positions
    bounds = ((0, 1),)*n  # multiplying a tuple makes n copies of it

    # constraint that weights need to equal 1
    weights_sum_to_1 = {
        'type': 'eq',
        # check (sum of weights) - 1 equals 0
        'fun': lambda weights: np.sum(weights) - 1
    }

    # neg sharpe ratio
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_returns(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov), method="SLSQP",
                       constraints=(weights_sum_to_1),
                       bounds=bounds)  # can pass arg: options = {'disp':False}
    return results.x
    
    
    

def plot_ef(n_points, er, cov, show_cml = False, riskfree_rate = 0, linestyle = ".-"):
    """
    Plots the N-asset efficient frontier
    """

    weights = optimal_weights(n_points, er, cov)

    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    # pack into df
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})

    # plot
    ax = ef.plot.line(x="Volatility", y="Returns", style = linestyle)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate,er,cov)
        r_msr = portfolio_returns(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Add Capital Markets Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate,r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = "o", linestyle = 'dashed')
        
    return ax
