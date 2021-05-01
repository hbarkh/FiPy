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
    hfi.index = hfi.index.to_period('m')
    
    hfi /= 100

    return hfi


def get_ind_size():
    """
    Load Ken French 30 Industry sizes
    """
    
    ind = pd.read_csv("data/ind30_m_size.csv",header=0,index_col=0,parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("m")
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ind_nfirms():
    """
    Load Ken French 30 Industry portfolio number of firms
    """
    
    ind = pd.read_csv("data/ind30_m_nfirms.csv",header=0,index_col=0,parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("m")
    ind.columns = ind.columns.str.strip()
    
    return ind


def get_ind_returns():
    """
    Load Ken French 30 Industry portfolios
    Value weighted monthly returns
    """
    
    ind = pd.read_csv("data/ind30_m_vw_rets.csv",header=0,index_col=0,parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("m")
    ind.columns = ind.columns.str.strip()
    
    return ind


def get_total_market_index_returns():
    """
    Get total market index returns by calculating the:
    number of firms in index, size of firms
    market cap, equal market cap weights, then return * equal cap weights
    """
    
    ind_returns = get_ind_returns()
    ind_nfirms = get_ind_nfirms()
    ind_size= get_ind_size()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 0)
    
    
    total_market_return = (ind_capweight * ind_returns).sum(axis=1)
    return total_market_return


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
    Modified refers to Cornish-Fisher Adjustment
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
    
def gmv(cov):
    """
    Returns the weights of the Global Minimum Vol portfolio
    Given the covariance matrix
    """
    n = len(cov)
    return msr(0, np.repeat(1,n), cov)
    

def plot_ef(n_points, er, cov, show_cml=False, show_ew=False, show_gmv = False, riskfree_rate=0, linestyle=".-"):
    """
    Plots the N-asset efficient frontier
    """

    weights = optimal_weights(n_points, er, cov)

    rets = [portfolio_returns(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    # pack into df
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})

    # plot
    ax = ef.plot.line(x="Volatility", y="Returns", style=linestyle)

    if show_ew:
        n = len(er)
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_returns(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        
        # Add EW portfolio
        ax.plot([vol_ew],[r_ew], color = 'goldenrod', marker = 'o')
        
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_returns(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        
        # Add EW portfolio
        ax.plot([vol_gmv],[r_gmv], color = 'royalblue', marker = 'o')

    if show_cml:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_returns(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Add Capital Markets Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker="o", linestyle='dashed')

    return ax


def run_cppi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, riskfree_rate = 0.01, drawdown = None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Return a dict containing: Asset Value History, Risk Budget History, Risk Weight History
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12

    # for backtesting:
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)

        cushion = (account_value - floor_value)/account_value

        risky_w = m * cushion    
        risky_w = np.minimum(risky_w, 1)  # no leverage
        risky_w = np.maximum(risky_w, 0)  # no shorting

        safe_w = 1 - risky_w

        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w

        # Update the account value for this time step
        account_value = risky_alloc*(1 + risky_r.iloc[step]) + safe_alloc*(1 + safe_r.iloc[step])

        # Save the values to plot and analyze
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
    
    
    risky_wealth = start * (1 + risky_r).cumprod()
    
    backtest_result = {
        "Wealth":account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation":risky_w_history,
        "m":m,
        "start":start,
        "floor":floor,
        "risky_r":risky_r,
        "safe_r":safe_r
    }
    
    return backtest_result

def summary_stats(r, riskfree_rate = 0.01):
    """
    Return a DataFrame that contains summary stats for the returns in the columns of r
    """
    
    ann_r = r.aggregate(annualize_rets,periods_per_year = 12)
    ann_vol = r.aggregate(annualize_vol,periods_per_year = 12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate = riskfree_rate, periods_per_year = 12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    
    result = pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Skewness": skew,
        "Kurtosis":kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic VaR (5%)": hist_cvar5,
        "Max Drawdown": dd,
    })
    
    return result
    
def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val
       

def discount(t,r):
    """
    :param t: time
    :param r: interest rate
    :return: (1+r)^(-t)
    """

    return (1+r)**(-t)

def pv(l,r):
    """
    Computes the present value of a sequence of liabilities
    :param l: is indexed by the time, and the values are the amounts of each liability
    :param r: interest rate
    :return: the present value of the sequence
    """
    dates = l.index
    discounts =discount(dates,r)

    return np.dot(discounts,l)

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rates
    :param assets: scalar of total assets value today
    :param liabilities: series
    :param r: interest rate
    :return: assets/pv(liabilities,r)
    """
    return assets/pv(liabilities,r)




