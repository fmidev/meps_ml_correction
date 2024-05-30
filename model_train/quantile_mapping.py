# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:05:59 2023

@author: ylinenk
"""
import numpy as np

##Quantile mapping for wind speed forecasts (not for forecasted error)

def interp_extrap(x, xp, yp):
    """      
    This function is used by q_mapping().
    Projects the x values onto y using the values of 
    [xp,yp], extrapolates conservatively if needed.
    """
    
    y = np.interp(x, xp, yp)
    y[x<xp[ 0]] = yp[ 0] + (x[x<xp[ 0]]-xp[ 0])
    y[x>xp[-1]] = yp[-1] + (x[x>xp[-1]]-xp[-1])
    return y

def q_mapping(obs,ctr,scn,variable,nq=10000,q_obs=False,q_ctr=False):
    """ 
    Quantile mapping. Three (n,) or (n,m) numpy arrays expected for input, 
        where n = time dimension and m = station index or grid cell index.
    
    First argument represents the truth, usually observations.
    
    Second is a sample from a model.
    
    Third is another sample which is to be corrected based on the quantile 
        differences between the two first arguments. Second and third 
        can be the same.
    
    Fourth is the number of quantiles to be used, i.e. the accuracy of correction.
    
    Previously defined quantiles can be used (fifth and sixth arguments), otherwise
        they are (re)defined 

    Linear extrapolation is applied if the third sample contains values 
        outside the quantile range defined from the two first arguments.

    Seventh argument is observation variable and it sets the limit for observations
        taken in to quantile training.
    """
    
    if q_obs:    
        return interp_extrap(scn,q_ctr,q_obs), q_obs, q_ctr
    
    else:    
        # Calculate quantile locations to be used in the next step
        q_intrvl = 100/float(nq); qtl_locs = np.arange(0,100+q_intrvl,q_intrvl) 

        #Pitää ehkä tehdä joku rajaus että kaikista äärevimmät 5-10 havaintoa poistetaan, koska ne muuten dominoivat liikaa
        #ind = np.argpartition(obs, -10)[-10:]
        #obs = np.delete(obs, ind)
        #ctr = np.delete(ctr, ind)
        if (variable == "windspeed"): ind = obs < 35
        if (variable == "windgust"): ind = obs < 45
        if (variable == "temperature"): ind = (obs > 233) & (obs < 313)
        if (variable == "dewpoint"): ind = (obs > 233) & (obs < 305)
        obs = obs[ind]
        ctr = ctr[ind]
        
        # Calculate quantiles
        q_obs = np.nanpercentile(obs, list(qtl_locs), axis=0)
        q_ctr = np.nanpercentile(ctr, list(qtl_locs), axis=0) 
        
        return interp_extrap(scn,q_ctr,q_obs), q_obs, q_ctr

