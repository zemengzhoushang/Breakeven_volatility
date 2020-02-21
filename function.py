#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:11:36 2020

@author: air
"""


from scipy.stats import norm
from math import log,sqrt,exp
import scipy.optimize as opt
import numpy as np



def bs_call(sigma,s,k,T,r):
    d1 = (log(s/k)+(r+0.5*sigma*sigma)*T)/sigma/sqrt(T)
    d2 = d1-sigma*sqrt(T)
    p = s*norm.cdf(d1)-k*exp(-r*T)*norm.cdf(d2)
    return p




def get_Gamma(s,k,T,r,sigma):
    d1 = (log(s/k)+(r+0.5*sigma*sigma)*T)/sigma/sqrt(T)
    return norm.pdf(d1)/s/sigma/sqrt(T)




def get_hisvol(s):
    log_r = []
    for i in range(1,len(s)):
        log_r.append(log(s[i]/s[i-1]))
    log_r_1 = np.array(log_r)
    st = np.std(log_r_1,ddof=1)
    
    return st*sqrt(252)




def get_impvol(marketprice,s,k,T,r):
    impvol = []
    for i in range(0,len(k)):
        f = lambda x: bs_call(x,s,k[i],T[i],r)-marketprice[i]
        impvol.append(opt.bisect(f,0,2))
        
    return impvol


def get_impvol_t(marketprice,s,k,T,r):
    impvol = []
    for i in range(0,len(k)):
        f = lambda x: bs_call(x,s,k[i],T,r)-marketprice[i]
        impvol.append(opt.bisect(f,-1,1))
        
    return impvol


def get_PnL_1(sigma,k,T,s,r):
    
    pnl = [None]*len(T)
    delta = [None]*len(T)
    b = [None]*len(T)
    
    delta[0] = norm.cdf((log(s[0]/k)+(r+0.5*sigma*sigma)*T[0])/sigma/sqrt(T[0]))
    b[0] = bs_call(sigma,s[0],k,T[0],r) - delta[0] * s[0] 
    pnl[0] =b[0]
    
    for i in range(1,len(T)):
        if T[i]!= 0 :
            d1 = (log(s[i]/k)+(r + 0.5 * sigma * sigma) * T[i])/sigma/sqrt(T[i])
            delta[i] = norm.cdf(d1)
            b[i] = -(delta[i] - delta[i-1]) * s[i] + b[i-1]* exp(r*(T[i-1]-T[i]))
            pnl[i] = b[i]-b[i-1]

    payoff = max(s[len(T)-1]-k,0)
    b[-1] = b[-2]*exp(r*(T[len(T)-2]-T[len(T)-1])) + delta[-2]*s[len(T)-1] - payoff
    pnl[-1] = b[-1]-b[-2]
    return np.array(pnl).sum()
    #return pnl
        



def get_RobustPnL_1(sigma,k,T,s,r):
    pnl = [0]*len(T);
    pnl[0] = 0 # bs_call(sigma,s[0],k,T[0],r)
    #pnl[0] = m
    gamma = get_Gamma(s[0],k,T[0],r,sigma)
    for i in range(1,len(T)-1):
        ds = (s[i]-s[i-1])/s[i-1]
        pnl[i] =  0.5 * gamma * s[i] * s[i] * (ds*ds - sigma * sigma * (T[i-1]-T[i]))
        if T[i]!=0.0:
            gamma = get_Gamma(s[i],k,T[i],r,sigma)
            
    #pnl[-1] =-1 * max(s[len(T)-1]-k,0)
    return np.array(pnl).sum()



    
def get_PnL_t(sigma,k_r,T,s,r):
    
    pnl = [None]*len(T)
    delta = [None]*len(T)
    b = [None]*len(T)
    a_pnl = [None]*s.shape[1]
    
    
    for j in range(s.shape[1]):
        k = s[0,j]*k_r
        delta[0] = norm.cdf((log(s[0,j]/k)+(r+0.5*sigma*sigma)*T[0])/sigma/sqrt(T[0]))
        b[0] = bs_call(sigma,s[0,j],k,T[0],r) - delta[0] * s[0,j] 
        pnl[0] =b[0]
        
        for i in range(1,len(T)):
            if T[i]!= 0 :
                d1 = (log(s[i,j]/k)+(r + 0.5 * sigma * sigma) * T[i])/sigma/sqrt(T[i])
                delta[i] = norm.cdf(d1)
                b[i] = -(delta[i] - delta[i-1]) * s[i,j] + b[i-1]* exp(r*(T[i-1]-T[i]))
                pnl[i] = b[i]-b[i-1]
    
        payoff = max(s[len(T)-1,j]-k,0)
        b[-1] = b[-2]*exp(r*(T[len(T)-2]-T[len(T)-1])) + delta[-2]*s[len(T)-1,j] - payoff
        pnl[-1] = b[-1]-b[-2]
        a_pnl[j] = (np.array(pnl).sum())
    
    return np.array(a_pnl).sum()
    #return pnl

   
     
def get_RobustPnL_t(sigma,k,T,s,r):
    pnl = [0]*len(T);
    pnl[0] = 0 # bs_call(sigma,s[0],k,T[0],r)
    a_pnl = [None]*s.shape[1]
    #pnl[0] = m
    for j in range(0,s.shape[1]):
        gamma = get_Gamma(s[0,j],k,T[0],r,sigma)
        for i in range(1,len(T)-1):
            ds = (s[i,j]-s[i-1,j])/s[i-1,j]
            pnl[i] =  0.5 * gamma * s[i,j] * s[i,j] * (ds*ds - sigma * sigma * (T[i-1]-T[i]))
            if T[i]!=0.0:
                gamma = get_Gamma(s[i,j],k,T[i],r,sigma)
        a_pnl[j] = np.array(pnl).sum()
    #pnl[-1] =-1 * max(s[len(T)-1]-k,0)
    return np.array(a_pnl).sum()
        
        
        












        
        
        

        
        
