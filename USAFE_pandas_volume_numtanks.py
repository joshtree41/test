# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:35:13 2018

@author: jgirardi
"""


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import sys
import re


def die():
    sys.exit()
#die() 
def cov(x):
    out = np.std(x)/np.mean(x)
    return out
def expfunc(t,a,b):
    return a*np.exp(b*t)
def powerfunc(t, m, c, c0):
    return c0 + t**m * c
def logfunc(t,a,b):
    return a*np.log(t)+b
def linfunc(t,a,b):
    return a*t+b
def poly3(t,a,b,c,d):
    return a*t**3 + b*t**2 + c*t + d
def poly4(t,a,b,c,d,e):
    return a*t**4 + b*t**3 + c*t**2 + d*t + e

def main():
    ## read in sql linked GADs 1.2.6 fuel data
    df_dims = pd.read_csv('ngafuel_USAFE.csv')
    NGAmap = pd.read_csv('NGAmap.csv')
    

###### check countries
#    check_AAFIF = df_dims.groupby(['country'], as_index = False)['R'].agg(np.size)
##    check_AAFIF = check['country']
##    print(check_AAFIF)
#    check_NGA = pd.DataFrame(columns = ['country'], data =np.array(['Norway', 'Sweden', 'Iceland', 'United Kingdom', 'Denmark',
#       'Estonia', 'Latvia', 'Poland', 'Lithuania', 'Germany',
#       'Netherlands', 'France', 'Belgium', 'Luxembourg', 'Czech Republic',
#       'Slovakia', 'Romania', 'Bulgaria', 'Hungary', 'Croatia', 'Slovenia',
#       'Italy', 'Portugal', 'Spain', 'Montenegro', 'Turkey', 'Greece',
#       'Albania']))
##    print(check_NGA)
#    decode = pd.read_csv('cc_decode.csv')
##    print(decode.head(10))
#    check_NGA = pd.merge(check_NGA, decode, left_on = 'country', right_on = 'country', how = 'left')
##    print(check_NGA)
#    outer = pd.merge(check_AAFIF, check_NGA, left_on = 'country', right_on= 'cc', how = 'outer')
#    outer.rename(columns = {'country_x': 'AAFIF_cc', 'country_y': 'NGA_cc', 'cc':'NGA_cc_decoded'}, inplace = True)
#    outer.drop(columns = 'R', inplace = True)
#    outer = pd.merge(outer, decode, left_on='AAFIF_cc', right_on='cc', how = 'left')
#    print(outer)
######
###### I've come to the conclusion that the two country sets are not significantly different
###### The decoder is incomplete though, so this isnt a very thurough check
    
###### done checking countries    
    ## compute volume
    df_dims['volume'] = np.pi*df_dims.R**2*df_dims.H*7.48
    ## compute tank level flag for zero-height anamoly
    df_dims['flag'] = np.where(df_dims.H == 0, 1, 0)
    
    ## sum over flag to identify bases that have at least one zero-height tank
###### total_H0_subset identifies bases with zero-height tanks
    total_H0_subset = df_dims.groupby('wac', as_index = False)['flag'].agg(np.sum)
    ## flagv2 only used for H0_subset to desiplay break up of 44 bases with zero-height tanks
    total_H0_subset['flagv2'] = np.where(total_H0_subset.flag > 0, 1, 0)
    ## breakout of bases with and without zer-height tanks
#    H0_subset = total_H0_subset.groupby('flagv2')['wac'].agg(np.size)
#    print(H0_subset)

    H0 = total_H0_subset.loc[total_H0_subset.flag > 0]
#    print(H0)
    H0_v2 = pd.merge(H0[['wac']], df_dims[['wac', 'R', 'H']], on = 'wac')
#    print(H0_v2)
#    H0_v2 = H0_v2.loc[H0_v2.H == 0]
#    print(H0_v2)
    

###### extapolating to calculate H based of R     
    
    ## identify the correct subset of data to make predictions based upon
    df_RH = df_dims[['wac', 'R', 'H']].loc[df_dims.flag == 0]
    x_bound = df_RH['R'].quantile(0.99)
    y_bound = df_RH['H'].quantile(0.99)
#    print(y_bound)
#    print(x_bound)
    df_RH = df_RH.loc[(df_RH['H'] <= y_bound)]
    df_RH = df_RH.loc[(df_RH['R'] <= x_bound)]
#    df_RH.plot.scatter(x = 'R', y = 'H')
    x = np.array(df_RH['R'])
    y = np.array(df_RH['H'])
    
    ## best fit is a 3rd degree polynomail, funcitons listed above
    funk = poly3
    f, fcov  = scipy.optimize.curve_fit(funk, x,  y,  maxfev=2000)
#    ## plot results and compute R^2
    xp = np.linspace(0, 160, num=320)
    ## viz with regression
    viz = plt.plot(x, y, '.', xp, funk(xp, *f),'-', label = 'f')
    plt.xlabel('R', fontsize=16)
    plt.ylabel('H', fontsize=16)
    ## R^2 caulculations
    residuals = y - funk(x, *f)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res/ss_tot)
#    print(r_squared)
    
    ## need to concatenate this information to the NGA_USAFE file
    H0_v2['H_extrap'] = funk(H0_v2.R, *f)
    H0_v2['H_final'] = np.where(H0_v2.H > 0, H0_v2.H, H0_v2.H_extrap)
    H0_v2['volume'] = (H0_v2.R**2)*(H0_v2.H_final)*np.pi*7.48
#    print(H0_v2)
    

###### continue with extrapolation of volume/numtanks/dims
    
    ## grab the set of wacs that do not have any zero-height tanks 
    total_H0_subset = total_H0_subset.loc[total_H0_subset.flag == 0]
    
    ## get aggreagtate stats from df_dims
    df = df_dims.groupby(['wac', 'cat', 'sector', 'country'])['volume'].agg([np.sum, np.size])
    ##formatting
    df.reset_index(col_level=0, inplace = True)
    df.rename(columns ={'sum': 'volume', 'size': 'numtanks'}, inplace = True)
    
    ## filter out df accoridng ot which wacs have no zero-height tanks     
    df = pd.merge(df, total_H0_subset[['wac']], on = 'wac')
    
    ## read in base runway data
    df_rwys = pd.read_csv('base_runways.csv')    
    ## merge input with runway data
    df = pd.merge(df, df_rwys, on='wac')
    
    
###### checking to see which of my wacs correspond to base survey wacs
    ## check to see which set of H0 bases are valid in the base survey
    ## total -> 44
    ## rwy >= 69000 -> 37
    ## merged with base survey... 36
    H0_rwy_check = pd.merge(df_rwys.loc[df_rwys.maxlen >= 6900], H0[['wac']], on = 'wac')
    df_survey_wacs = pd.read_csv('survey_wacs.csv')
    H0_rwy_check = pd.merge(H0_rwy_check, df_survey_wacs, on = 'wac', how = 'inner')
#    print(H0_rwy_check)
    H0_rwy_check = H0_rwy_check[['wac']]
#    print(H0_rwy_check)
#    compare = pd.merge(total_H0_subset, df_survey_wacs, on = 'wac', how = 'left')
#    compare = pd.merge(df_rwys.loc[df_rwys.maxlen >= 6900], compare, on = 'wac')
#    print(compare)

    
    
    ## parametrize runway max length
    df['maxlen_cat'] = np.where(df['maxlen']<7000, 'less than 7000', \
    np.where(df['maxlen']<8000, '7000-8000', np.where(df['maxlen']<9000, \
    '8000-9000', np.where(df['maxlen']<10000,'9000-10000', '<10000'))))
    ##change units to millions
    df['volume'] = df['volume'].apply(lambda x: x/1000000)

    
    ## this will  be used instead of the regression as the relationship between volume and number of tanks
    df['bin_volume_upper'] = \
    np.where(df['volume']<1, 1, \
    np.where(df['volume']<2, 2, \
    np.where(df['volume']<4, 4, \
    np.where(df['volume']<6, 6, \
    np.where(df['volume']<10, 10, \
    np.where(df['volume']<15, 15, 20))))))
    df = df.loc[df['volume']<=50]
    
    ## trim very obvious outliers - over 40 tanks
    df_tanks = df[df['numtanks']<40]
    df_tanks = df.groupby(df['bin_volume_upper'])['numtanks'].agg([np.mean, np.size, np.std])
    df_tanks.reset_index(col_level=0, inplace = True)
    df_tanks.rename(columns ={'mean': 'numtanks'}, inplace = True)    
    df_tanks['numtanks'] = np.round_(df_tanks['numtanks'])    


    ## categorical determination of volume
    ## can this be done in a for loop?
    ## by categopry code
    df_by_cat = df.groupby(['cat'])['volume'].agg([np.mean, np.std, np.size])
    df_by_cat.reset_index(col_level = 0, inplace = True)
    df_by_cat['cov_cat'] = df_by_cat['std'] / df_by_cat['mean']
    df_by_cat.rename(columns = {'mean': 'average_fuel_by_cat'}, inplace = True)
#    print(df_by_cat)
    
    df_by_maxlen = df.groupby(['maxlen_cat'])['numtanks'].agg([np.mean, np.std, np.size])
    df_by_maxlen.reset_index(col_level = 0, inplace = True)
    df_by_maxlen['cov_maxlen'] = df_by_maxlen['std'] / df_by_maxlen['mean']
    df_by_maxlen.rename(columns = {'mean': 'average_fuel_by_maxlen'}, inplace = True)
#    print(df_by_maxlen)
    
    df_by_numrwys = df.groupby(['numrwys'])['numtanks'].agg([np.mean, np.std, np.size])
    df_by_numrwys.reset_index(col_level = 0, inplace = True)
    df_by_numrwys['cov_numrwys'] = df_by_numrwys['std'] / df_by_numrwys['mean']
    df_by_numrwys.rename(columns = {'mean': 'average_fuel_by_numrwys'}, inplace = True)
#    print(df_by_numrwys)
    
    df_by_country = df.groupby(['country'])['volume'].agg([np.mean, np.std, np.size])
    df_by_country = df_by_country.loc[df_by_country['std'].notnull()]
    df_by_country.reset_index(col_level = 0, inplace = True)
    df_by_country['cov_country'] = df_by_country['std'] / df_by_country['mean']
    df_by_country.rename(columns = {'mean': 'average_fuel_by_country'}, inplace = True)
#    print(df_by_country)
        
    ## join each of the aggreagetes, chose the one with the smallest cov
    
    df_89 = pd.read_csv('input_91.csv')
    df_89 = pd.merge(df_89, df_rwys, on='wac')
    df_89['maxlen_cat'] = np.where(df_89['maxlen']<7000, 'less than 7000', \
    np.where(df_89['maxlen']<8000, '7000-8000', np.where(df_89['maxlen']<9000, \
    '8000-9000', np.where(df_89['maxlen']<10000,'9000-10000', '<10000'))))
#    print(df_89)    

    ## merge wacs for extrapolation by each of the characteristics
    df_extapolation = pd.merge(df_89, df_by_cat[['cat', 'average_fuel_by_cat','cov_cat']], on = ['cat'])
#    print(df_extapolation)
    df_extapolation = pd.merge(df_extapolation, df_by_maxlen[['maxlen_cat', 'average_fuel_by_maxlen', 'cov_maxlen']], on = ['maxlen_cat'])
#    print(df_extapolation)
    df_extapolation = pd.merge(df_extapolation, df_by_numrwys[['numrwys', 'average_fuel_by_numrwys', 'cov_numrwys']], on = ['numrwys'])
#    print(df_extapolation)
    df_extapolation = pd.merge(df_extapolation, df_by_country[['country', 'average_fuel_by_country', 'cov_country']], on = ['country'], how = 'left')
#    print(df_extapolation)

    ## create columns that stor the best choice by characteristic name and the covariance associated with it
    df_extapolation['best_cov'] = df_extapolation[['cov_cat', 'cov_maxlen', 'cov_numrwys', 'cov_country']].min(axis =1)
    df_extapolation['best_stor'] = df_extapolation[['cov_cat', 'cov_maxlen', 'cov_numrwys', 'cov_country']].idxmin(axis =1)
    df_extapolation['best_stor'] = df_extapolation['best_stor'].apply(lambda x: x[4:])
#    print(df_extapolation)
    df_writeup = df_extapolation.groupby(['best_stor'])['best_cov'].agg(np.size)
#    print(df_writeup)    
    ## extract the volume assiciated with the best choice according to COV
    ## list will hold values
    temp_list = []
    for i in range(len(df_extapolation)):
        ## extract category
        best = str(df_extapolation['best_stor'][i])
        ## us category to extract volume assocaited with category
        best2 = df_extapolation['average_fuel_by_' + best][i]
        ## build up the list
        temp_list.append(best2)
    ## assign volume field in extrapalation df to be equal to the list created above
    df_extapolation['volume']=np.array(temp_list)
    ## create a bin for estiamted volume so that this can be linked back in order to extract the number of tanks
    df_extapolation['bin_volume_upper'] = \
    np.where(df_extapolation['volume']<1, 1, \
    np.where(df_extapolation['volume']<2, 2, \
    np.where(df_extapolation['volume']<4, 4, \
    np.where(df_extapolation['volume']<6, 6, \
    np.where(df_extapolation['volume']<10, 10, \
    np.where(df_extapolation['volume']<15, 15, 20))))))
    
    ## merge the extrapolated volume with the dataframe that reslates numtanks to voulme
    df_extapolation = pd.merge(df_extapolation, df_tanks, on = 'bin_volume_upper')
    df_extapolation['volume'] = df_extapolation['volume']*1000000*0.1337
    subset = df_extapolation[['wac', 'best_stor', 'volume', 'numtanks']]
    subset['numtanks'] = pd.to_numeric(subset['numtanks'], downcast = 'integer')
#    print(subset)
    
    ## computing R:H in order to backcalculate dimensions
    RtoH = df_dims['R'].sum(axis=0) / df_dims['H'].sum(axis=0)
#    print(RtoH)
    
    
    ## viz for Mike
#    df_dims_v2 = df_dims[['R', 'H', 'volume']]
#    threshold = df_dims_v2['volume'].quantile(0.25)
#    df_dims_v3 = df_dims_v2.loc[df_dims_v2.H != 0].loc[df_dims_v2.volume <= threshold]
#    df_dims_v3.plot.scatter(x = 'R', y = 'volume')

    ## compute dimensions
    subset['H'] = (subset.volume/(subset.numtanks*np.pi*(RtoH)**2))**(1/3)
    subset['R'] =  subset.H*RtoH
    subset['check'] = np.pi*(subset.R)**2*subset.H*subset.numtanks
#    print(subset[['volume', 'check']])
    
    df_out = subset[['wac', 'numtanks', 'R', 'H', 'volume']]
    df_out['volume'] = df_out['volume']*7.48
    df_out['volume_tank'] =  df_out['volume'] / df_out['numtanks']
    df_out = df_out.loc[df_out.index.repeat(df_out['numtanks'])].reset_index(drop=True)
    df_out = df_out[['wac','R','H', 'volume_tank']]
#    print(df_out)
    
####### 91 exrapolated bases
#    df_out.to_csv('df_extrapolation_91_new.csv', index=False)
####### 36 bases in suvey with zero heigh tanks
    H0_v2 = pd.merge(df_survey_wacs, H0_v2, on = 'wac')
    H0_v2 = pd.merge(H0_v2, NGAmap, left_on = 'wac', right_on = 'wac_innr')
#    print(H0_v2)
#    print(len(H0_v2.groupby('wac').agg(np.size))) ---> 36 correct
#    H0_v2.to_csv('df_extrapolation_36_H0.csv')
###### 36 bases unique + rwy data
#    H0_rwy_check.to_csv('df_extrapolation_36_H0_uniqueWACs.csv')
    
       
main()

def main2():
    NGAmap = pd.read_csv('NGAmap.csv')
    print(NGAmap.head(10))
    
#main2()