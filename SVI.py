#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:49:16 2020

@author: air
"""

from scipy.optimize import minimize
import numpy as np

def svi_param(msigma,f,adc,strikes,m_vol,maturity):
    m, sigma = msigma
    sigma = max(0, sigma)


    def outter_fun(msigma):
        
        global a_star, d_star, c_star
        
        def inner_fun(adc):
       
            """
            内层函数 用残差最小 拟合估计参数a d c，slsqp 
            注意对implied vol 进行转换 成 omega=vol**2*t
            """
            a,d,c = adc
            error_i_sum = 0.0
            xi = np.log(strikes/f) # rate
            y = (xi-m)/sigma
            z = np.sqrt(y**2+1)
            error_i_sum = np.sum(np.array(a + d * y + c * z -np.array(m_vol)**2* maturity) ** 2)
            
            return error_i_sum
        '''
        cons = (
                    {'type': 'ineq', 'fun': lambda x:x[2]},
                    {'type': 'ineq', 'fun': lambda x:x[2]-abs(x[1])},
                    {'type': 'ineq', 'fun': lambda x:4*sigma-x[2]-abs(x[1])},
                    {'type': 'ineq', 'fun': lambda x:x[0]},
                    {'type': 'ineq', 'fun': lambda x:max(m_vol)-x[0]}
                    )
         '''
        cons = (
            {'type': 'ineq', 'fun': lambda x: x[2]-abs(x[1])},
            {'type': 'ineq', 'fun': lambda x: 4*sigma-x[2]-abs(x[1])}
        )               
        inner_res = minimize(inner_fun, adc, method='SLSQP', tol=1e-6,constraints = cons)
        a_star, d_star, c_star = inner_res.x
    
    
        error_o_sum = 0.0
        xi = np.log(strikes/f)
        y = (xi-m)/sigma
        z = np.sqrt(y**2+1)
        error_o_sum = np.sum(np.array(a_star + d_star * y + c_star *
                              z - np.array(m_vol)**2*maturity) ** 2)
        return error_o_sum
        
    outter_res = minimize(outter_fun,msigma,method='Nelder-Mead', tol=1e-6)
    m_star, sigma_star = outter_res.x
    
    #obj = outter_res.fun
    calibrated_params = [m_star, sigma_star,a_star,d_star,c_star]
    return calibrated_params


def svi_vol(f,strikes,param,maturity):
   
    m_star,sigma_star,a_star,d_star,c_star = param
    xi = np.log(strikes/f)
    y = (xi-m_star)/sigma_star
    z = np.sqrt(y**2+1)
   
    omega = np.array(a_star + d_star * y + c_star * z)
    sigma = np.sqrt(omega/maturity)

    return sigma