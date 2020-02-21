#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:39:18 2020

@author: air
"""

from jqdatasdk import *
import pandas as pd
import numpy as np
import os
import re
import datetime
import function as func
from math import log,sqrt,exp
import matplotlib.pyplot as plt
import scipy.optimize as opt
from itertools import chain
import SABR
import SVI
import Heston
import dill
import time


os.getcwd()
os.chdir('/Users/air/Desktop/project')
auth('13701531168','531168')
dill.load_session('variable.pkl')

#Data collection and manipulation
get_security_info('510050.XSHG').name #上证ETF50
etf = get_price('510050.XSHG','2015-02-05','2020-02-05','daily')
option_price = pd.read_csv('option_price.csv')
option_info = pd.read_csv('option_info.csv')

option_c_price  = option_price.loc[option_price['Option_Name'].str.contains('购')]
option_c_info = option_info.loc[option_info['Option_Name'].str.contains('购')]
option_c_info['exercise_date'] = pd.to_datetime(option_c_info.exercise_date)
option_c_info['expire_date'] = pd.to_datetime(option_c_info.expire_date)
option_c_price.reset_index(inplace=True)
option_m = pd.merge(option_c_price,option_c_info,how='left',on=['option_ID','Option_Name'])
option_m[option_m['exercise_date'].isna()].Option_Name


#Time transform
option_m.drop(columns=['Trading_ID_x','Trading_ID_y'],inplace=True)
option_m['Date']=pd.to_datetime(option_m.Date)
option_m['exercise_date'] = pd.to_datetime(option_m.exercise_date)
option_m['expire_date'] = pd.to_datetime(option_m.expire_date)

option_m['time_to_maturity'] = option_m['exercise_date']-option_m['Date']
sum(option_m[option_m['status']==1]['time_to_maturity'].isna())

k_min = option_c_info['strike'].min() #2.05
k_max = option_c_info['strike'].max() #3.5


etf.reset_index(inplace=True)
etf.rename(columns={'index':'Date'},inplace=True)
etf['Date'].dt.strftime('%Y-%m-%d') 

delta_min = option_m['Delta'].min()
delta_max = option_m['Delta'].max()

#data analysis
name_info = set(option_c_info['Option_Name']) #251
len(name_info)
name_price = set(option_c_price['Option_Name']) #328
len(name_price)
name_diff = name_price - name_info
len(name_diff)

pattern = re.compile('\d?\d?\d?\d?年?[0-9]+月')
pattern.findall(option_c_price['Option_Name'][9])
expire_date = []
for i in range(option_c_price.shape[0]):
    expire_date.append(pattern.findall(option_c_price['Option_Name'][i]))

expire_date_1  = list(chain(*expire_date))
#expire_date = pd.DataFrame(expire_date).transpose()

date_map={'2019年2月':'2019-02-27',
          '2019年3月':'2019-03-27',
          '2019年4月':'2019-04-24',
          '2019年5月':'2019-05-22',
          '2019年6月':'2019-06-26',
          '2019年7月':'2019-07-24',
          '2019年8月':'2019-08-28',
          '2019年9月':'2019-09-25',
          '2019年10月':'2019-10-23',
          '2019年11月':'2019-11-27',
          '2019年12月':'2019-12-25',
          '1月':'2020-01-22',
          '2月':'2020-02-26',
          '3月':'2020-03-25',
          '6月':'2020-06-24',
          '9月':'2020-09-23',
          '2020年1月':'2020-01-22',
          '2020年2月':'2020-02-26',
          '2020年3月':'2020-03-25',
          '2020年6月':'2020-06-24',
          '2020年9月':'2020-09-23',
          }

expire_date_2 = [date_map[x] if x in date_map else x for x in expire_date_1]
option_c_price['expire_date'] = pd.to_datetime(expire_date_2)
option_c_price['Date'] = pd.to_datetime(option_c_price.Date)
option_c_price['time_to_maturity'] = option_c_price['expire_date']-option_c_price['Date']
option_c_price['time_to_maturity'] = option_c_price['time_to_maturity']/datetime.timedelta(days=252)

option_c_price_r = option_c_price.iloc[::-1]
option_c_price_r.drop('index',axis = 1,inplace=True)
option_c_price_r.reset_index(inplace=True)
option_c_price_r.drop('index',axis = 1,inplace=True)
option_c_price_r.sort_values(by=['Date','expire_date','strike'],ascending=True,inplace=True)
option_c_price_r.reset_index(inplace=True)
option_c_price_r.drop('index',axis=1,inplace=True)



"""""""""integration"""""""""
"""""""""Time_stamp: 2019-02-11"""""""""

time_stamp = ['2019-02-11',
              '2019-03-05',
              '2019-04-04',
              '2019-05-06',
              '2019-06-05',
              '2019-07-05',
              '2019-08-05',
              '2019-09-05',
              '2019-10-08',
              '2019-11-05',
              '2019-12-05',
              '2020-01-06']

temp_vol = np.linspace(0.1,0.9,9)

for a in range(0,len(time_stamp)):
    t_s = pd.to_datetime(time_stamp[a])
    temp = option_c_price_r[option_c_price_r['Date']==t_s]
    s_1 = float(etf[etf['Date']==t_s]['close'])
    expire = pd.to_datetime(pd.unique(temp[temp['Date']==t_s]['expire_date']))
    expire = expire[expire<pd.to_datetime('2020-02-05')]
    expire
    
    ind = etf[etf['Date']==t_s].index.tolist()
    temp_bev = etf[ind[0]:etf.shape[0]]
    temp_bev.reset_index(drop=True,inplace=True)
    
    spot_hist = etf['close'][max(ind[0]-252,0):ind[0]]
    spot_hist.reset_index(drop=True,inplace=True)
    hv = func.get_hisvol(spot_hist)
    
    
    fig = plt.figure(figsize=(12,12))
    for i in range(0,len(expire)):
        #i = 3
        Now = str(t_s).strip(' 00:00:00')
        Expire = str(expire[i]).strip(' 00:00:00')
        #days = str(expire[i]-t_s).strip(' 00:00:00')
        title = Now + ' to '+Expire#+ '  '+ days
        
        k_1 = temp[temp['expire_date']==expire[i]]['strike']
        k_1.reset_index(drop=True,inplace=True)
        T_1 = temp[temp['expire_date']==expire[i]]['time_to_maturity']
        T_1.reset_index(drop=True,inplace=True)
        marketprice = temp[temp['expire_date']==expire[i]]['close']
        marketprice.reset_index(drop=True,inplace=True)
        imp_vol = func.get_impvol(marketprice,s_1,k_1,T_1,r) 
    
        k_bev = np.linspace(k_1.min(),k_1.max(),len(k_1))
        T_bev = (expire[i]- temp_bev['Date'])/datetime.timedelta(days=252)
        T_bev = T_bev[T_bev>=0]
        #bev_brent = []
        #bev_bisect=[]
        bev_delta=[]
        bev_gamma=[]

        #bev_2 = []
        for i_s in range(len(k_bev)):
            temp_newton = []
            f = lambda x: func.get_PnL_1(x,k_bev[i_s],T_bev,temp_bev['close'],r)
            #f_1 = lambda x: func.get_RobustPnL(x,k_bev[i],T_bev,temp_bev['close'],r)
            #bev_brent.append(opt.brentq(f,-1,5))
            #bev_bisect.append(opt.bisect(f,-1,5))
            for j in range(0,9):
                try:
                    temp_v = opt.newton(f,temp_vol[j])
                except RuntimeError:
                    continue
                else:
                    temp_newton.append(temp_v)
            if len(temp_newton)!=0:
                bev_delta.append(min(temp_newton,key=lambda x:abs(x-hv)))
            else:
                bev_delta.append(opt.brentq(f,-1,5))
                  
        for j_s in range(len(k_bev)):
            temp_newton_1 = []
            f_1 = lambda x:func.get_RobustPnL_1(x,k_bev[j_s],T_bev,temp_bev['close'],r)
            #f_2 = lambda x: func.get_RobustPnL(x,k_bev[i],T_bev,temp_bev['close'],r)
            #bev_brent.append(opt.brentq(f,-1,5))
            #bev_bisect.append(opt.bisect(f,-1,5))
            for j in range(0,9):
                try:
                    temp_v_1 = opt.newton(f_1,temp_vol[j])
                except RuntimeError:
                    continue
                else:
                    temp_newton_1.append(temp_v_1)
            if len(temp_newton_1)!=0:
                bev_gamma.append(min(temp_newton_1,key=lambda x:abs(x-hv)))
            else:
                bev_gamma.append(opt.brentq(f_1,-1,5))
            
         
            #bev_1.append(opt.brentq(f_1,-5,5))
            #bev_2.append(opt.brentq(f_2,-1,1))
        '''   
        bev_1 = pd.DataFrame([k_bev,bev_brent]).transpose()
        bev_1.rename(columns={0:'strike',1:'bev_brent'},inplace=True)
        bev_1 = bev_1[bev_1['bev_brent']>0.001]
        bev_1.reset_index(drop=True,inplace=True)
    
        
        bev_2 = pd.DataFrame([k_bev,bev_bisect]).transpose()
        bev_2.rename(columns={0:'strike',1:'bev_bisect'},inplace=True)
        bev_2 = bev_2[bev_2['bev_bisect']>0.001]
        bev_2.reset_index(drop=True,inplace=True)
        '''
        bev_3 = pd.DataFrame([k_bev,bev_delta]).transpose()
        bev_3.rename(columns={0:'strike',1:'bev_delta'},inplace=True)
        bev_3 = bev_3[bev_3['bev_delta']>0.001]
        bev_3.reset_index(drop=True,inplace=True)       
        
        bev_4 = pd.DataFrame([k_bev,bev_gamma]).transpose()
        bev_4.rename(columns={0:'strike',1:'bev_gamma'},inplace=True)
        bev_4 = bev_4[bev_4['bev_gamma']>0.001]
        bev_4.reset_index(drop=True,inplace=True)    
        
        
        ax = fig.add_subplot(2,2,i+1)
        plt.tight_layout()
        ax.plot(k_1/float(s_1),imp_vol,color = 'green',label = 'imp_vol')
        #ax.plot(bev_1['strike']/float(s_1),bev_1['bev_brent'],color = 'red',label = 'bev_brent')
        #ax.plot(bev_2['strike']/float(s_1),bev_2['bev_bisect'],color = 'orange',label = 'bev_bisect')
        ax.plot(bev_3['strike']/float(s_1),bev_3['bev_delta'],color = 'brown',label = 'bev_delta')
        ax.plot(bev_4['strike']/float(s_1),bev_4['bev_gamma'],color = 'red',label = 'bev_gamma')
        ax.plot(k_1/float(s_1),[hv]*len(k_1),color = 'blue',label = 'histvol')
        ax.legend()
        ax.set_xlabel('strike%')
        ax.set_ylabel('volatility')
        ax.set_title('%s' %title)
    Now = Now+'_r'
    plt.savefig('/Users/air/Desktop/%s.jpg' %Now )

""""""""""""""""""""""""

"""""""""Sanity Check"""""" """
r=0.0
s0 = 2.7
sigma = 0.25
T = 0.083
n = 30
k = np.linspace(s0-0.25*sigma,s0+0.25*sigma,10)
p = []
for i in range(0,10):
    p.append(func.bs_call(sigma,s0,k[i],T,r))

#generate price path
#np.random.seed(13)
s = np.zeros([31,16])
maturity = np.zeros([31,1])
s[0,:] = s0
maturity[0] = T
dt = T/n

i_s=0
flag = 0
while i_s < etf.shape[0] -30:
    close = etf['close'][i_s:i_s+31]
    s[:,flag]=close
    i_s = i_s+30
    flag = flag +1
    

#delta hedging
fig = plt.figure(figsize=(12,12))
fig.suptitle('Delta')
#for a in range(0,9):
for j in range(0,s.shape[1]):
    for i in range(1,s.shape[0]):
        p1 = (r-0.5*sigma*sigma)*dt
        p2 = sigma*np.sqrt(dt)
        s[i,j] =s[i-1,j]*exp(p1+p2*np.random.normal(0,1))
        maturity[i] = T - i*dt
               
    bev_brent_1=[]
    bev_bisect_1=[]
    error = []
    for i in range(len(k)):
        f = lambda x: func.get_PnL_t(x,k[i],maturity,s,r)
        #f_1 = lambda x: func.get_PnL_1(x,k_bev[i_s],T_bev,temp_bev['close'],r)
        bev_brent_1.append(opt.newton(f,sigma))
        
        #bev_bisect_1.append(opt.bisect(f,-1,5))
    error = abs((np.array(bev_brent_1) - np.array([sigma]*len(k)))/sigma*100)
    
    ax1 =fig.add_subplot(3,3,a+1)         
    fig.tight_layout()
    ax1.plot(k/s[0,0],bev_brent_1,color='red',label='bev')
    #plt.plot(k/s[0,0],bev_bisect_1,color='orange')
    ax1.plot(k/s[0,0],[sigma]*len(k),color = 'orange',label='sigma')
    ax1.legend()
    ax1.set_ylim([0.2,0.4])
    ax1.set_xlabel('strike%')
    ax1.set_ylabel('volatility')
    
    ax2 = ax1.twinx()
    ax2.plot(k/s[0,0],error,'blue',label = 'error%')
    ax2.legend(loc='upper left')
    ax2.set_ylim([0,5])
    ax2.set_ylabel('error%')

plt.savefig('/Users/air/Desktop/Delta_0.3.png')
#a = a+1

#gamma hedging
fig = plt.figure(figsize=(16,12))
fig.suptitle('Gamma')
for a in range(0,9):
    for j in range(0,s.shape[1]):
        for i in range(1,31):
            p1 = (r-0.5*sigma*sigma)*dt
            p2 = sigma*np.sqrt(dt)
            s[i,j] =s[i-1,j]*exp(p1+p2*np.random.normal(0,1))
            maturity[i] = T - i*dt
    
    
    bev_brent_1=[]
    bev_bisect_1=[]
    error = []
    for i in range(len(k)):
        f = lambda x: func.get_RobustPnL_t(x,k[i],maturity,s,r)
        #f_1 = lambda x: func.get_PnL_1(x,k_bev[i_s],T_bev,temp_bev['close'],r)
        #f_2 = lambda x: func.get_RobustPnL(x,k_bev[i],T_bev,temp_bev['close'],r)
        #bev_brent_1.append(opt.brentq(f,0.0001,1))
        bev_bisect_1.append(opt.newton(f,sigma))
    error = abs((np.array(bev_bisect_1) - np.array([sigma]*len(k)))/sigma*100)
    
    ax1 = fig.add_subplot(3,3,a+1)
    fig.tight_layout()
    #plt.plot(k/s[0],bev_brent_1,color='red',label='bev_brent')
    ax1.plot(k/s[0,0],bev_bisect_1,color='red',label='bev')
    ax1.plot(k/s[0,0],[sigma]*len(k),color = 'orange',label='sigma')
    ax1.legend()
    ax1.set_ylim([0.2,0.5])
    ax1.set_xlabel('strike%')
    ax1.set_ylabel('volatility')
    
    ax2 = ax1.twinx()
    ax2.plot(k/s[0,0],error,'blue',label = 'error%')
    ax2.legend(loc='upper left')
    ax2.set_ylim([0,4])
    ax2.set_ylabel('error%')

plt.savefig('/Users/air/Desktop/Gamma_0.25.png')
a = a+1
    


i_a = 0
while i_a < etf.shape[0]-30:
    close = etf['close'][i_a:i_a+30]
    close.reset_index(drop=True,inplace=True)
    hvol = func.get_hisvol(close)
    T =  np.linspace(0,30/252,30)
    T_1 = []
    for i in reversed(list(T)):
        T_1.append(i)
    T_1 = np.array(T_1)
    b1 = []
    b2 = []
    b3 = []
    k_t = np.linspace(close[0]*0.8,close[0]*1.2,10)
    for i in range(len(k_t)):
        temp_newton_1=[]
        f_t = lambda x:func.get_PnL_1(x,k_t[i],T_1,close,r)
        #b1.append(opt.brentq(f_t,-1,1))
        for j in range(0,9):
            try:
                temp_v_1 = opt.newton(f_t,temp_vol[j])
            except RuntimeError:
                continue
            else:
                temp_newton_1.append(temp_v_1)
        b2 .append(min(temp_newton_1,key=lambda x:abs(x-hvol)))
   # plt.plot(k_t/close[0],b1,'r',label = 'bev_b')
    plt.plot(k_t/close[0],b2,'orange',label='bev_n')
    plt.plot(k_t/close[0],[hvol]*len(k_t),'blue',label='hisvol')
    plt.legend()
    plt.xlabel('strike%')
    plt.ylabel('volatility')
    title = str(int(i_a/30))
    plt.savefig('/Users/air/Desktop/%s.jpg'%title)
    plt.show()
    i_a=i_a+30

    

""""""""""""""""""""""""

"""Get Breakeven volatility """

param_sabr=np.zeros([4,3])
param_svi =np.zeros([4,5])

etf = get_price('510050.XSHG','2015-02-05','2020-02-05','daily')
etf.reset_index(inplace=True)
etf.rename(columns={'index':'Date'},inplace=True)
fig = plt.figure(figsize=(12,6))
plt.plot(etf['Date'],etf['close'])
plt.xlabel('Date')
plt.ylabel('Price')


r=0.0
n = 30
title = str(n)+'days'
T =  n/365#one month
s = np.zeros([n+1,int(etf.shape[0]/n)])
maturity = [0]*s.shape[0]
maturity[0] = T
dt = T/n

for i in range(s.shape[0]):
    maturity[i] = T - i*dt
maturity[-1] = 0.0

i_s=0
flag = 0
while i_s < etf.shape[0] -n:
    close = etf['close'][i_s:i_s+n+1]
    s[:,flag]=close
    i_s = i_s+n
    flag = flag +1
    
temp_vol = np.linspace(0.1,0.9,5)

hisv = func.get_hisvol(etf['close'][0:i_s])
k = np.linspace(0.8,1.2,20)



bev=[]
error = []
for i in range(len(k)):
    temp_newton = []
    f = lambda x: func.get_PnL_t(x,k[i],maturity,s,r)
    #f_1 = lambda x: func.get_RobustPnL_t(x,k[i],maturity,s,r)
    for j in range(len(temp_vol)):
        try:
            temp_v = opt.newton(f,temp_vol[j])
        except RuntimeError:
            continue
        else:
            temp_newton.append(temp_v)
    bev.append(min(temp_newton,key=lambda x:abs(x-hisv)))
    
'''
bev_30 = bev
bev_60 = bev
bev_90 = bev
bev_120=bev
'''

    #bev_bisect_1.append(opt.bisect(f,-1,5))
error = abs((np.array(bev_brent_1) - np.array([hisv]*len(k)))/hisv*100)

fig = plt.figure(figsize=(12,12))   
ax1 =fig.add_subplot(1,1,1)    
ax1.plot(k,bev,color='red',label='bev')
#plt.plot(k/s[0,0],bev_bisect_1,color='orange')
ax1.plot(k,[hisv]*len(k),color = 'orange',label='hisvol')
#ax1.plot(k,error,color='brown',label='error')
ax1.legend()
#ax1.set_ylim([0.2,0.4])
ax1.set_ylim([0,1])
ax1.set_xlabel('strike')
ax1.set_ylabel('volatility')
ax1.set_title(title)
plt.savefig('/Users/air/Desktop/%s.jpg'%title)
""""""""""""""""""

"""Fit Breakeven volatility """
        
"""""""""SABR Model"""""""""
#A good fit can be obtained for any 0<=beta<=1. Often beta=0,1/2 or beta=1
#is choosen, depending on the market.
beta = 0.5
sabr_result = np.zeros([s.shape[1],3])
start_1 = time.clock()
for i in range(s.shape[1]):
    #The current forward price
    f =  s[0,i]*np.exp(r*maturity[0])
    sigmas = bev
    #The 'At the money volatility', corrosponding to a strike equal to the current forward price.
    atm_sigma = bev[9]
    #An inital guess of the parameters alpha, nu and rho.
    guess = [0.01, 10,-0.5]
    #calculating the actual strikes from f and strikes_in_bps
    strikes = f*k
    #Calling the SABR_calibration function defined below to return the parameters.
    alpha, nu, rho = SABR.SABR_calibration(f, T, atm_sigma, beta, strikes, sigmas, guess)
    sabr_result[i,:] = np.array([alpha,nu,rho])
    
    vols_from_sabr = SABR.SABR_market_vol(strikes,f,T,alpha,beta,nu,rho)
    fig, ax = plt.subplots()
    plt.plot(k, sigmas, 'x',label='Actual_bev')
    plt.plot(k,vols_from_sabr,label='SABR_bev')
    plt.xlabel("Strikes in bps")  
    plt.ylabel("Market volatilities")
#plt.text(0.6, 0.9, textbox, transform=ax.transAxes, fontsize=10,
#    verticalalignment='top',bbox=dict(facecolor='white', alpha=0.7))
#plt.savefig(f"{t_exp} year into {tenor} year swaption"+".png")     

alpha, nu, rho = np.mean(sabr_result,axis = 0)
f = np.mean(s[0,:])*np.exp(r*maturity[0])
strikes = f*k
vols_from_sabr = SABR.SABR_market_vol(strikes,f,T,alpha,beta,nu,rho)
end_1 = time.clock()
run_time_1 = end_1-start_1
print('sabr_running_time: ',run_time_1)
param_sabr[0,:]=np.array([alpha, nu, rho])
error = abs(vols_from_sabr - bev)/np.array(bev)*100
max_error_1 = max(error)


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(k, sigmas, 'x',label='Actual_bev')
ax1.plot(k,vols_from_sabr,color='red',label='SABR_bev')
ax1.legend()
ax1.set_xlabel("Strikes%")
ax1.set_xlim([0.78,1.22])
ax1.set_ylabel("Volatilities")
ax1.set_title('SABR_Model')

ax2 = ax1.twinx()
ax2.plot(k,error,'--',color='orange',label = 'error%')
ax2.legend(loc='upper left')
ax2.set_ylabel('error%')
ax2.set_ylim([0,10])

plt.savefig('/Users/air/Desktop/SABR_Model_%s'%title)

""""""""""""""""""""""""

"""""""""SVI model"""""""""
svi_result = np.zeros([s.shape[1],5])
start_2=time.clock()
for i in range(s.shape[1]):
    param = []
    msigma = [0.1,hisv]
    adc = [0.02,0.02,-1]
    f = s[0,i]*np.exp(r*maturity[0])
    strikes = f*k
    a_star, d_star, c_star = [0,0,0]
    
    param = SVI.svi_param(msigma,f,adc,strikes,bev,T)
    svi_result[i,:] = np.array(param)
    
    vols_from_svi = SVI.svi_vol(f,strikes,param,T)
    fig, ax = plt.subplots()
    plt.plot(k, sigmas, '+',label='Actual_bev')
    plt.plot(k,vols_from_svi,label='SVI_bev')
    plt.legend()
    plt.xlabel("Strikes%")
    plt.ylabel("Volatilities")
    
param = np.mean(svi_result,axis = 0)
f = np.mean(s[0,:])*np.exp(r*maturity[0])
strikes = f*k
vols_from_svi = SVI.svi_vol(f,strikes,param,T)
end_2 = time.clock()
run_time_2 = end_2-start_2
print('svi running time: ', run_time_2)
param_svi[0,:] = np.array(param)
error = abs(vols_from_svi - bev)/np.array(bev)*100
max_error_2 = max(error)

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(k, bev, 'x',label='Actual_bev')
ax1.plot(k,vols_from_svi,color='red',label='SVI_bev')
ax1.legend()
ax1.set_xlabel("Strikes%")
ax1.set_ylabel("Volatilities")
ax1.set_xlim([0.78,1.22])
ax1.set_title('SVI_Model')

ax2 = ax1.twinx()
ax2.plot(k,error,'--',color='orange',label = 'error%')
ax2.legend(loc='upper left')
ax2.set_ylabel('error%')
ax2.set_ylim([0,10])

plt.savefig('/Users/air/Desktop/SVI_Model_%s'%title)


""""""""""""""""""""""""
filename = 'variable.pkl'
dill.dump_session(filename)