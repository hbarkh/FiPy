import pandas as pd
import numpy as np
import scipy.stats as ss


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
