from __future__ import print_function
import pandas as pd
import numpy as np
import math
import scipy
from time import time 
import datetime as dt
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
from pandas import datetime
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, freqz, filtfilt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VARMAX
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors 
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from reportlab.lib.units import inch

# import math 
# from scipy import stats
# plt.rcParams['figure.figsize'] = [16, 12]
# plt.rcParams.update({'font.size': 18})
# from statsmodels.tsa.arima_model import ARMA
# import statsmodels.tsa.stattools as ts
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import grangercausalitytests
# from statsmodels.formula.api import ols
# from statsmodels.tsa.tsatools import lagmat
# from scipy.stats import pearsonr

def load():
    print('Loading kiln data...')
    t1 = time()
    df = pd.read_csv('Full_Kiln_Data_2015-2020b.csv') 
    df  = df.set_index('datetime')
    df = df.drop(df.index[-1])
    df.index = pd.to_datetime(df.index)

    df2 = pd.read_csv('Tags_and_Units.csv')
    new_cols = df2.columns.values
    new_cols[0] = 'Variable'
    df2.columns = new_cols
    df2 = df2.set_index('Variable')

    Y_And = [1449, 1432, 1410, 1379, 1360,  1327, 1232, 1188, 1127, 1090, 1077, 1041, 
            1021, 1004, 993, 987, 918, 904, 882, 860, 827, 770, 757, 738]

    X_And = [15, 20, 30, 37, 43, 50, 60, 80, 100, 113, 120, 133, 150, 154, 157, 160, 
            195, 210, 220, 230, 240, 249, 254, 259]
    
    df = df[~df.index.duplicated(keep='first')]
    
    t2 = time()
    
    print('Loading time is', t2-t1, 'seconds')
    
    return df, Y_And, X_And, df2


def hyper():
    print('Loading hyperparameters...')
    t1 = time()

    h_clean = [12, 0.01, 0.0001, 120]
    # h_clean[0] = 12 - minimum time 
    # h_clean[1] = 0.01 - Absolute difference limit accounts for static/frozen sensors
    # h_clean[2] = 0.0001 - Standard deviation limit accounts for frozen sensors that drift.
    # h_clean[3] = 120 - Mean limit accounts for shutdown data where the sensors may still
    #  output readings with variations

    h_startup = [] # Still need to fill this one out.
    h_benchmark = []

    # Two hmap params below need to have same length.
    h_hmap = [1, 1, 1, 0, 0, 0, 0] # Control which heatmaps are offered and interpolated
    # Benchmark methods are at indices 2 to 5

    h_hmaplist = ["Raw","Clean","Trials","Benchmark-Scheduled","Benchmark-Nov2015",
                "Benchmark-Apr2016","Benchmark-Sep2016"] 

    h_trial = [720] # h_trial[0] = cutoff for min trial length
    h_trialstudy = []
    #h_framestudy = [60,300,600,60] # step_size, min_win_size, max_win_size, prediction horizon
    h_framestudy = [12,300,600,60] # step_size, min_win_size, max_win_size, prediction horizon
    
    # Currently it is setup to only model on 1st endog var. Fix this!
    #h_endog = ['KST_50','KST_60','KST_80']
    h_endog = ['KST_60']
    h_exog = ['KST_15','KST_20','KST_30','KST_37','KST_43','KST_50','KST_80','KST_100','KST_113','KST_120','KST_133','KST_150',
               'KST_154','KST_157','KST_160','KST_195','KST_210','KST_220','KST_230','KST_240','KST_249','KST_254','KST_259']
               #'NG_2K','KF','KO','KFS']
    #h_exog = ['KST_15','KST_20','KST_30']#,'KST_20','KST_30','KST_37','KST_100','KST_113','KST_240','KF','KO','KFS','NG_2K']
    h_framecols = h_endog+h_exog
    '''
    h_framecols = ['1_FAS','M21F','1_MFV','2_FAS','M22F','2_MFV','KS_MF_T',
            'KF','FDE_T','K_FDE_T','ID_FI_T','LKS','WLS_T','K_FDE_O2','KO','K_FRE_TR',
            'K_FRE_TC','KFS','BMZ_T','FMZ_T','NG_2K','CACO3_T','WL_T_pS','LQ_EST','1_CST',
            'WL_T_pCE','C1_pCE','W_S','A_T','KST_15','KST_20','KST_30','KST_37','KST_43',
            'KST_50','KST_60','KST_80','SHS_0_90','K_LB_S','K_Beam', 'KST_195', 'KST_210',
            'KST_220', 'KST_230', 'KST_240', 'KST_249', 'KST_254', 'KST_259', 'Gearbox', 
            'SHS_0_180', 'Dump_Gate', 'KST_100','KST_113','KST_120', 'KST_133','KST_150', 
            'KST_154','KST_157', 'KST_160','WW_TTA']
    '''
    h_preproc = [15,1/3600,1/30000,75,70]

    # Study using mod = ar_select_order(d_tst[13].KST_60,maxlag=60) determined use of either
    # two lags [1, 2] or first seven lags. Empirical results disagree.
    h_model = [[1,0]] # h_model[0] = ARX-lags
    
    d = {'h_clean': h_clean, 'h_startup': h_startup, 'h_benchmark': h_benchmark, 
        'h_hmap': h_hmap, 'h_hmaplist': h_hmaplist, 'h_trial': h_trial,
        'h_trialstudy': h_trialstudy, 'h_framestudy': h_framestudy, 'h_endog': h_endog, 
        'h_exog':h_exog, 'h_framecols': h_framecols, 'h_preproc':h_preproc, 'h_model':h_model} 

    # Return as dataframe
    d_hyper = pd.DataFrame.from_dict(d, orient='index').T
    
    t2 = time()

    print('Loading time is', t2-t1, 'seconds')

    return d_hyper


# Improvements to make
        # if nan values are less than h_clean[3] = 5 consecutive hours then simple ffill that data.
        # This should apply to auxiliary variables too.
def clean(df,h_clean):
    print('Cleaning kiln data...')
    t1 = time()
    data_columns = df.columns
    df = (df.drop(data_columns, axis=1).join(df[data_columns].apply(pd.to_numeric, errors='coerce')))
    
    # New cleaning. Eventually this should be for all variables. But I am going to start with KST.
    # h_clean = [minimum time, sum of differences threshold, std-dev of differences threshold, mean threshold]

    for j in range(len(df.iloc[:,0:24].columns)):
        for i in range(len(df)):
            if (df.iloc[i:i+h_clean[0],j].diff().abs().sum() <= h_clean[1] or 
                df.iloc[i:i+h_clean[0],j].diff().std() <= h_clean[2]):
                df.iloc[i,j] = np.nan
    
    for i in range(len(df)):
        if (df.iloc[i,0:24].mean() <= h_clean[3] and df.iloc[i+1,0:24].mean() <= h_clean[3]):
            df.iloc[i,0:24] = np.nan

    # Previous cleaning by camera
    '''
    for i in range(len(df)):
        # Camera 1
        if (df.iloc[i:i+2,0:8].diff().abs().sum().sum() <= h_clean[0] or 
            df.iloc[i:i+8,0:8].diff().std().sum() <= h_clean[1]):
            df.iloc[i,0:8] = np.nan
        # Camera 2
        if (df.iloc[i:i+2,8:16].diff().abs().sum().sum() <= h_clean[0] or 
            df.iloc[i:i+8,8:16].diff().std().sum() <= h_clean[1]):
            df.iloc[i,8:16] = np.nan
        # Camera 3
        if (df.iloc[i:i+2,16:24].diff().abs().sum().sum() <= h_clean[0] or 
            df.iloc[i:i+8,16:24].diff().std().sum() <= h_clean[1]):
            df.iloc[i,16:24] = np.nan
        # All 3 cameras
        if (df.iloc[i,0:24].mean() <= h_clean[2] and df.iloc[i+1,0:24].mean() <= h_clean[2]):
            df.iloc[i,0:24] = np.nan
    '''
          
    t2 = time()
    d_clean = df
    print('Cleaning time is', t2-t1, 'seconds')
    return d_clean


# Improvements to make
    # This whole function could use serious revision as I simply ported an old workflow
    # in with minimal changes as I was focused on getting a minimum working example.
    # Need to define h_startup and actually use it.
def startup(df1,df2,h_startup):
    print('Collecting startup profiles of kiln data...')
    t1 = time()
    # The first extraction of reference profiles was performed on raw data df1 and
    # revised on clean data df2.
    df_Ref = df1.iloc[2:4,0:24]
    #need to drop first two rows afterwards
    count = 2
    df_Ref['Index'] = ""
    Avg_len = 96
    delay = 96

    for i in range(len(df1)):
        if ((df1.KF.iloc[i:i+6].mean() <= 20 or df1.KF.iloc[i:i+6].diff().std() <= 0.001) and
            df1.iloc[i,0:24].mean() <= 65):
            for j in range(240):
                if (df1.iloc[i+j:i+j+Avg_len,0:24].mean(axis=1).mean() > 160 and 
                    df1.iloc[i+j:i+j+Avg_len,0:24].mean(axis=1).diff().mean() <= 1.0 and 
                    df1.KF.iloc[i+j:i+j+Avg_len].mean() > 320 and 
                    df1.iloc[i+j:i+j+Avg_len,0:24].mean().isna().sum() == 0 and 
                    df1.iloc[i+j+delay:i+j+Avg_len+delay,0:24].mean().isna().sum() == 0):
                    p = i+j+delay
                    #print(p)
                    df_Ref = df_Ref.append(df1.iloc[i+j+delay:i+j+Avg_len+delay,0:24].mean(),
                                            ignore_index=True)
                    #df_Ref = df_Ref.append(df.iloc[i+j:i+j+Avg_len,0:24].mean(), ignore_index=True)
                    df_Ref.Index[count] = df1.index[p]
                    count = count+1
                    break 
            
    df_Ref = df_Ref.iloc[2:]
    df_Ref = df_Ref.drop_duplicates()
    df_Ref = df_Ref.set_index('Index')
    
    # Switching to after data has been cleaned to revise the startup profiles
    df_Ref['Index'] = ""
    df_Ref.Index[0] = df2.index[4309]
    for i in range(len(df_Ref)):
        j = df2.index.get_loc(df_Ref.index[i])
        print(j)
        if df2.iloc[j:j+Avg_len].mean().isna().sum() == 0:
            df_Ref.iloc[i] = df2.iloc[j:j+Avg_len,0:24].mean()
            df_Ref.Index[i] = df2.index[j]
        elif df2.iloc[j:j+Avg_len].mean().isna().sum() <= 24:
            for k in range(Avg_len):
                if df2.iloc[j+k:j+Avg_len+k].mean().isna().sum() == 0:
                    df_Ref.iloc[i] = df2.iloc[j+k:j+Avg_len+k,0:24].mean()
                    p = j+k
                    df_Ref.Index[i] = df2.index[p]
                    break
        else:
            df_Ref.iloc[i] = np.nan
    
    df_Ref = df_Ref.set_index('Index')
    df_Ref.index = pd.to_datetime(df_Ref.index)
    
    t2 = time()        
    print('Startup profile collection time is', t2-t1, 'seconds')
    
    return df_Ref


# Improvements to make. 
    # Use h_benchmark to indicate which benchmark profiles to load.
    # This still needs to be made to work and actually return something.
def benchmark(df,df_Ref,Y_And,X_And,h_benchmark):
    print('Generating benchmark profiles...')
    t1 = time()
        
    gamma = [abs(30 - x) for x in X_And[2:21]]
    end = [0, 0, 0]
    beg = [1, 1]
    gamma = 1 - np.array(gamma)/210
    gamma = np.array(beg + list(gamma) + end)
    TG_star =  Y_And
    
    TG_sched = []
    k = 0
    for i in range(len(df)):
        TS_star = df_Ref.iloc[k]
        if k <= 8 and i >= df.index.get_loc(df_Ref.index[k+1]):
            k = k+1
            TS_star = df_Ref.iloc[k]
        delta = TG_star/TS_star
        TG_sched.append((df.iloc[i,0:24].values+((TS_star[2]-df.iloc[i,2])*gamma+(TS_star[20]-df.iloc[i,20])*(1-gamma)))*delta.values)

    TG_sched = pd.DataFrame(TG_sched, columns = df.columns[0:24])
    E_sched = TG_star - TG_sched
    E_sched = E_sched.set_index(df.index)

    TG_Nov2015 = []
    for i in range(len(df)):
        TS_star = df_Ref.iloc[0]
        delta = TG_star/TS_star
        TG_Nov2015.append((df.iloc[i,0:24].values+((TS_star[2]-df.iloc[i,2])*gamma+(TS_star[20]-df.iloc[i,20])*(1-gamma)))*delta.values)

    TG_Nov2015 = pd.DataFrame(TG_Nov2015, columns = df.columns[0:24])
    E_Nov2015 = TG_star - TG_Nov2015
    E_Nov2015 = E_Nov2015.set_index(df.index)

    # Additional benchmark error signals can be added (see Appendix Z of Nov 5 code)
    d_benchmarks = {}
    df_list = [E_sched,E_Nov2015]
    df_list_names = ['E_sched','E_Nov2015']
    for i in range(len(df_list)):
        d_benchmarks[df_list_names[i]] = df_list[i]
    # Run d_benchmarks.keys() to get the list of dictionary keys
    # For subscriptable values use list(d_benchmarks)[0] 

    t2 = time()        
    print('Benchmark profile generation time is', t2-t1, 'seconds')
    return d_benchmarks


def trial(df, df2, d_hyper):
    t1 = time()
    print('Collecting trial runs')
    # This code creates a new column titled Good, which indicates whether there any any nan values in the 
    # KST values of each row. If there aren't any nan values then Good = 1 otherwise Good = 0
    cutoff = d_hyper.h_trial[0] #cutoff = 720 # 120 for 5 days or 720 for 30 days. This could be a hyper-param but prob too meta.

    df["Good"] = ""
    for i in range(len(df)):
        if df.iloc[i,0:24].isna().sum().sum() == 0:
            df['Good'].iloc[i] = 1
        else:
            df['Good'].iloc[i] = 0

    # This code denotes start and end times for slicing the datetime index to lump consecutive rows of Good data together.
    start = []
    end = []

    for i in range(len(df)):
        if df.Good.diff().iloc[i] == 1:
            start.append(i)
        elif df.Good.diff().iloc[i] == -1:
            end.append(i)
    
    # j will store the indices of start and end that are of interest for filtering data
    j = []
    for i in range(len(start)):
        if end[i]-start[i] >= cutoff:
            j.append(i)                
    
    df = df.drop(['Good'], axis=1)
    
    d_bigsamp = pd.DataFrame(columns=df2.columns,index=df2.index)
    d = {}
    for i in range(len(j)):
        d[i] = df.iloc[start[j[i]]:end[j[i]]]
        d_bigsamp.loc[d[i].index[0]:d[i].index[-1]] = d[i]
            
    t2 = time()        
    print('Data sampling time is', t2-t1, 'seconds')
    return d, d_bigsamp


# Thermal camera interpolation
def interp(df):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    t1 = time()
        
    new_index=np.arange(15,260)
    
    cols = ['KST_15','KST_20','KST_30','KST_37','KST_43','KST_50','KST_60','KST_80', 
            'KST_100','KST_113','KST_120','KST_133','KST_150','KST_154','KST_157', 'KST_160',
            'KST_195', 'KST_210', 'KST_220', 'KST_230', 'KST_240', 'KST_249','KST_254','KST_259']

    X_And = [15, 20, 30, 37, 43, 50, 60, 80, 100, 113, 120, 133, 150, 154, 157, 160,
             195, 210, 220, 230, 240, 249, 254, 259]

    df = df[cols]
    df.columns = X_And
    df = df.T.astype('float')
    
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    df_out = df_out.astype(float)
    
    t2 = time()        
    print('Interpolation time is', t2-t1, 'seconds')
    
    return df_out


# Improvements to make
    # This code controls which interpolated heatmaps are created. 
    # Once all possible hmaps are known I should condense this code by vectorizing it into a pre-specified list
    # for i in len(h_hmap) ... if h_hmap[i] == 1: return_list.append(d_list[i]) ... Same for f_hmaplist
def hmaps(d_raw, d_clean, d_bigsamp, d_benchmarks, h_hmap):
    print('Generating interpolated heatmaps...')
    t1 = time()
    return_list = []
    
    if h_hmap[0] == 1:
        d_raw_int = interp(d_raw)
        return_list.append(d_raw_int)
    if h_hmap[1] == 1:
        d_clean_int = interp(d_clean)
        return_list.append(d_clean_int)
    if h_hmap[2] == 1:
        d_bigsamp_int = interp(d_bigsamp)
        return_list.append(d_bigsamp_int)
    if h_hmap[3] == 1:
        d_benchsched = d_benchmarks['E_sched']
        d_benchsched_int = interp(d_benchsched)
        return_list.append(d_benchsched_int)
    if h_hmap[4] == 1:
        d_benchNov15 = d_benchmarks['E_Nov2015']
        d_benchNov15_int = interp(d_benchNov15)
        return_list.append(d_benchNov15_int)
    # Can continue if h_hmap[5] == 1, for adding more benchmark heatmaps.
    
    t2 = time()        
    print('Batch interpolation time is', t2-t1, 'seconds')
    return return_list


def hmaplist(h_hmap, h_hmaplist):
    d_hmaplist = []
    for i in range(len(h_hmap)):
        if h_hmap[i] == 1:
            d_hmaplist.append(h_hmaplist[i])   
    
    return d_hmaplist


# Improvements to make
        # Add dropdown for KST values. Remove some useless H_map options
# Interactive heatmap with PV dropdown
def edit_heatmap(y_slider, m_slider, d_slider, h_slider, w_slider, PV, H_map, Transform):
    
    DF = hmaps[0]

    # This code accounts for the manual selector for year/month/day/hour
    dt = pd.DataFrame({'year': y_slider,
                   'month': m_slider,
                   'day': d_slider,
                   'hour': h_slider}, index=[0])
    dt = pd.to_datetime(dt)
    # We find the row # of df that matches the value of dt (from the user)
    j = DF.T.index.get_loc(dt.iloc[0])

    SMALL_SIZE = 18     #Font sizes
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize= MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # Set the heatmap based on the dropdown selector (H_map)
    if H_map == 'Raw' :
        DF = hmaps[0]
    elif H_map == 'Clean':
        DF = hmaps[1]
    elif H_map == 'Trials':
        DF = hmaps[2]
    elif H_map == 'Benchmark-Scheduled' :
        DF = hmaps[3]
    elif H_map == 'Benchmark-Nov2015' :
        DF = hmaps[4]
    # More elifs can be added to add more heatmap options. Check the Nov5_2020.ipynb for details

    if Transform == 'Raw':
        DF = DF
        #d_pv = d_raw
    elif Transform == 'Difference':
        DF = DF.T.diff().T
    elif Transform == 'Diff_12h_Sum':
        DF = DF.T.diff().rolling(12).sum().T
    elif Transform == 'Diff_24h_Sum':
        DF = DF.T.diff().rolling(24).sum().T
    elif Transform == 'Diff_48h_Sum':
        DF = DF.T.diff().rolling(48).sum().T
    # More elifs can be added for more Transform options.
    
    fig, ax = plt.subplots(2,figsize=(30,15))
    plt.style.use("ggplot")
    im = ax[0].imshow(DF.iloc[:,j:j+w_slider],aspect = 'auto', cmap='magma', vmin=30, vmax = 330)
    cbar = ax[0].figure.colorbar(im, ax=ax)#[0])     # Create colorbar
    cbar.set_label(label="Temperature ($^\circ$C)", rotation=-90, va="bottom", fontsize=18)

    # THIS CODE IS FOR THE HEATMAP
    y_m = [z*0.3048 for z in DF.index]
    ax[0].tick_params(axis='both', direction='out')
    ax[0].set_xticks([0,int((w_slider)/4),int((w_slider)/2),int((w_slider)*3/4),w_slider-1])
    ax[0].set_xticklabels([DF.columns[j],DF.columns[j+int((w_slider)/4)],DF.columns[j+int((w_slider)/2)],DF.columns[j+int((w_slider)*3/4)],DF.columns[j+w_slider-1]])    
    ax[0].set_xlabel("Date")
    #ax[0].set_yticks([0,60,120,180,240])  
    #ax[0].set_yticklabels([DF.index[0], DF.index[60], DF.index[120], DF.index[180], DF.index[240]])
    ax[0].set_yticks([0, int(np.round(len(DF)/4,0)), 2*int(np.round(len(DF)/4,0)), 
        3*int(np.round(len(DF)/4,0)), len(DF)]) 

    ax[0].set_yticklabels([np.round(y_m[0],1), np.round(y_m[0+int(np.round(len(DF)/4,0))],1), 
        np.round(y_m[0+2*int(np.round(len(DF)/4,0))],1), np.round(y_m[0+3*int(np.round(len(DF)/4,0))],1), np.round(y_m[-1],1)])
    
    
    
    ax[0].set_ylabel("Position from firing end of kiln (m)")
    
    # Need to re-assign df (below) so that it corresponds to the heatmap profile and updated data
    # THIS CODE IS FOR THE DROPDOWN BAR. INSTEAD OF d_raw IT NEEDS TO BE IF ELIF TO SELECT VERSION OF PVs. VERSION = Raw, VERSION = Preprocess,
    # VERSION = Compare --> benchmark to measured to forecast.... etc. They all need to have same columns names
    # Currently PV is d_tags.columns.to_list() but this needs to be investigated.
    
    ln1, = ax[1].plot(pd.Series(data=raw[tags[PV].loc['Short']].values[j:j+w_slider]), label = PV + '\n' + tags[PV].loc['Tag'])
    #ln1, = ax[1].plot(pd.Series(data=d_pv[d_tags[PV].loc['Short']].values[j:j+w_slider]), label = PV + '\n' + d_tags[PV].loc['Tag'])
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].legend(handles=[ln1],loc="upper right")
    ax[1].set_xlabel("Date")
    ax[1].set_xticks([0,int((w_slider)/4),int((w_slider)/2),int((w_slider)*3/4),w_slider-1])
    ax[1].set_xticklabels([DF.columns[j],DF.columns[j+int((w_slider)/4)],DF.columns[j+int((w_slider)/2)],DF.columns[j+int((w_slider)*3/4)],DF.columns[j+w_slider-1]])
    ax[1].set_ylabel(PV + ' (' + tags[PV].loc['Unit'] + ')')
    
    return plt.show()


def Prelim_Vis(d_hyper,d_tags,d_hmaps,d_raw):
    d_hmaplist = hmaplist(d_hyper.h_hmap,d_hyper.h_hmaplist)
    #['Raw', 'Scheduled', 'Nov2015', 'Outlier', 'Butterworth','Outlier-Butterworth'],
    # 'Apr2016', 'Sep2016', 'June2017', 'Apr2018', 'Oct2018', 'Feb2019', 'Oct2019', 'Mar2020', 'Aug2020']
    Transform_list = ['Raw', 'Difference', 'Diff_12h_Sum', 'Diff_24h_Sum', 'Diff_48h_Sum']

    y_slider = widgets.IntSlider(min=2015,max=2020,value=2019, step=1, description='year')
    m_slider = widgets.IntSlider(min=1,max=12,value=5, step=1, description='month')
    w_slider = widgets.IntSlider(min=12,max=45820,value=48, step=12, description='window size')
    d_slider = widgets.IntSlider(min=1,max=31,value=1, step=1, description='day')
    h_slider = widgets.IntSlider(min=0,max=24,value=1, step=1, description='hour')

    # MIGHT NEED TO EDIT PV here
    PV = widgets.Dropdown(options=d_tags.columns.tolist(), description = 'process variable')
    
    H_map = widgets.Dropdown(options=d_hmaplist, description = 'heatmap')
    Transform = widgets.Dropdown(options=Transform_list, description = 'transform')
    ui0 = widgets.VBox([y_slider,m_slider])
    ui1 = widgets.VBox([d_slider,h_slider])
    ui2 = widgets.VBox([w_slider,PV])
    ui3 = widgets.VBox([H_map,Transform])
    ui4 = widgets.HBox([ui0,ui1,ui2,ui3])

    # This global declaration was only necessary when I moved the code to VScode
    global hmaps, raw, tags
    hmaps = d_hmaps
    raw = d_raw
    tags = d_tags

    out = widgets.interactive_output(edit_heatmap, {'y_slider':y_slider,'m_slider':m_slider,
         'd_slider':d_slider,'h_slider':h_slider,'w_slider':w_slider,'PV':PV, 'H_map':H_map, 'Transform': Transform })

    display(ui4,out)


def trialstudy(d_trials, d_raw, d_hyper):
    t1 = time()
    print('Analyzing all trial runs')
    h_framecols = d_hyper.h_framecols
    h_framecols = [k for k in h_framecols if k] # Remove empty entries from framecols
    h_endog = d_hyper.h_endog
    h_endog = [k for k in h_endog if k] # Remove empty entries from framecols
    
    d_for = {}
    d_unc = {}
    d_mod = {}
    d_tst = {}
    # Forecasted predictions.
    d_forecast = pd.DataFrame(columns=h_endog,index=d_raw.index)
    # Extra forecast information.
    d_uncertain = pd.Series(index = d_raw.index)
    #d_modelfit = pd.Series(index=d_raw.index)
    d_modelsummary = pd.DataFrame(index=d_raw.index)
    # Test values to compare to forecasted predictions
    d_testunscaled = pd.DataFrame(columns=h_framecols,index=d_raw.index)
    
    for i in range(len(d_trials)):
        df = d_trials[i]

        df = df[~df.index.duplicated(keep='first')]
        df = df.asfreq(freq='1H')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

        df_iter = [i,len(d_trials)]
        
        ## Initiate frame-level analysis
        d_for[i], d_unc[i], d_mod[i], d_tst[i] = framestudy(df, df_iter, d_hyper)
        # These commands simply store the frame results in a big single dataframe.
        # I should probably just write a small function that does this and then remove these lines/variables
        d_forecast.loc[d_for[i].index[0]:d_for[i].index[-1]] = d_for[i]
        d_uncertain.loc[d_unc[i].index[0]:d_unc[i].index[-1]] = d_unc[i]
        d_testunscaled.loc[d_tst[i].index[0]:d_tst[i].index[-1]] = d_tst[i]

    t2 = time()
    Exp_t = round(t2-t1,1)
    print('Total trial analysis time is', t2-t1, 'seconds')
    return d_forecast, d_uncertain, d_testunscaled, d_for, d_unc, d_mod, d_tst, Exp_t


def framestudy(df, df_iter, d_hyper):
    t1 = time()
    h_framestudy = d_hyper.h_framestudy
    k = 0
    # i iterates in h_frame[0] increments between 0 and len(df) - max_win_size
    # Why max_win_size? Should probably be stepsize or forecast size. 
    for i in range(0,len(df),h_framestudy[0]):
        # If i+min_win_size is greater than max_win_size
        # Generally this block executes 2nd unless h[1]=h[2].
        if i+h_framestudy[1] >= h_framestudy[2]:
            while (k*h_framestudy[0]+h_framestudy[2]+h_framestudy[3]) <= len(df):
                # i_frame is the current iteration and the total number of iterations (if rounded down)
                i_frame = [i/h_framestudy[0], (len(df)-h_framestudy[2])/h_framestudy[0]]
                d_train = df.iloc[0+k*h_framestudy[0]:k*h_framestudy[0]+h_framestudy[2]]    
                d_test = df.iloc[k*h_framestudy[0]+h_framestudy[2]:k*h_framestudy[0]+h_framestudy[2]+h_framestudy[3]]
                d_pretrain, d_pretest, d_uncertain, mu, std = preproc(d_train, d_test, i_frame, d_hyper)
                d_model, exog = model(d_pretrain, i_frame, d_hyper)
                #d_train_rep = f_represent(I=I, df=d_train_preproc, )
                d_forecast, d_test_unscale = forecast(d_pretest, d_model, exog, mu, std, i_frame, d_hyper)
                endog_col = d_pretest.columns.difference(exog.columns)
                RMSE = np.sqrt(np.mean((d_forecast-d_test_unscale[endog_col])**2))
                #MAE = round(np.mean(abs(d_residuals/d_test)),4)
                print("The sliding window RMSE is", RMSE.values[0])
                
                k = k+1
                
                if i == 0:
                    d_rec_preproc = d_pretrain
                    d_rec_uncertain = d_uncertain
                    d_rec_forecast = d_forecast
                    d_rec_mod = {}
                    d_rec_mod[d_train.index[-1]] = d_model
                    d_rec_test = d_test_unscale

                d_rec_preproc = d_rec_preproc.append(d_pretrain.iloc[-1-h_framestudy[0]:-1])
                d_rec_uncertain = d_rec_uncertain.append(d_uncertain)
                d_rec_forecast = d_rec_forecast.append(d_forecast)
                d_rec_mod[d_train.index[-1]] = d_model
                d_rec_test = d_rec_test.append(d_test_unscale)
            
        else:
            i_frame = [i/h_framestudy[0], (len(df)-h_framestudy[2])/h_framestudy[0]]
            d_train = df.iloc[0:i+h_framestudy[1]]
            d_test = df.iloc[i+h_framestudy[1]:i+h_framestudy[1]+h_framestudy[3]]
            d_pretrain, d_pretest, d_uncertain, mu, std = preproc(d_train, d_test, i_frame, d_hyper)
            d_model, exog = model(d_pretrain, i_frame, d_hyper)
            d_forecast, d_test_unscale = forecast(d_pretest, d_model, exog, mu, std, i_frame, d_hyper)
            endog_col = d_pretest.columns.difference(exog.columns)
            RMSE = np.sqrt(np.mean((d_forecast-d_test_unscale[endog_col])**2))
            print("The growing window RMSE is", RMSE.values[0])
            
            if i == 0:
                d_rec_preproc = d_pretrain
                d_rec_uncertain = d_uncertain
                d_rec_forecast = d_forecast
                d_rec_mod = {}
                d_rec_mod[d_train.index[-1]] = d_model
                d_rec_test = d_test_unscale

            else: 
                d_rec_preproc = d_rec_preproc.append(d_pretrain.iloc[-1-h_framestudy[0]:-1])
                d_rec_uncertain = d_rec_uncertain.append(d_uncertain)
                d_rec_forecast = d_rec_forecast.append(d_forecast)
                d_rec_mod[d_train.index[-1]] = d_model
                d_rec_test = d_rec_test.append(d_test_unscale)

    t2 = time()
    print('Trial number', df_iter[0], 'analysis time is', t2-t1, 'seconds')
    return d_rec_forecast, d_rec_uncertain, d_rec_mod, d_rec_test #, d_record_uncertain#d_train, d_results


def f_1D_butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def f_1D_butter_lowpass_filter(data, cutoff, fs, order, padlen):
    b, a = f_1D_butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    z = filtfilt(b, a, data, axis=0, padlen=padlen)
    return y, z


def preproc(df, df1, i_frame, d_hyper):    
    t1 = time()
    h_framecols = d_hyper.h_framecols
    h_framecols = [k for k in h_framecols if k] # Remove empty entries from framecols
    # Should merge filter params below into hyper params
    order = 15
    #order = 12
    fs = 1/3600 # sampling frequency in Hz
    cutoff  = 1/30000 # in Hz
    b, a = f_1D_butter_lowpass(cutoff, fs, order)
    padlen = min((d_hyper.h_framestudy[3]-1),3*max(len(a),len(b)))
    
    # d_uncertain should only apply to h_framecols
    d_uncertain = pd.Series(data=df[h_framecols].isna().sum().sum(), index=df.index[-1:])
    # Need to add outliers to uncertainty measure. Need to add signal to noise ratio to uncertainty measure.
    #df = df.fillna(method='ffill')
    #print(df.isna().sum().sum())
    
    d_pretrain = pd.DataFrame(df[h_framecols], columns=h_framecols)
    d_pretest = pd.DataFrame(df1[h_framecols], columns=h_framecols)
    mu = pd.DataFrame(index=[0],columns=d_hyper.h_framecols.values.tolist())
    std = pd.DataFrame(index=[0],columns=d_hyper.h_framecols.values.tolist())
    
    for col in h_framecols:
        # Need a more universal approach to removing outlier data. 
        # The 75 deg C diff limit works for KST data but what about other vars?
        if col[0:3] == 'KST':            
            d_pretrain[[col]] = d_pretrain[[col]].where(d_pretrain[[col]].diff() < 75)
            d_pretrain[[col]] = d_pretrain[[col]].where(d_pretrain[[col]].diff() > -75)
            d_pretrain[[col]] = d_pretrain[[col]].where(d_pretrain[[col]] > 70)

            d_pretest[[col]] = d_pretest[[col]].where(d_pretest[[col]].diff() < 75)
            d_pretest[[col]] = d_pretest[[col]].where(d_pretest[[col]].diff() > -75)
            d_pretest[[col]] = d_pretest[[col]].where(d_pretest[[col]] > 70)
        
        d_pretrain[[col]] = d_pretrain[[col]].fillna(method='ffill')
        d_pretrain[[col]] = d_pretrain[[col]].fillna(method='bfill')
        d_pretest[[col]] = d_pretest[[col]].fillna(method='ffill')
        d_pretest[[col]] = d_pretest[[col]].fillna(method='bfill')        
        
        # Apply LP-filter to each col
        data = d_pretrain[col].values
        y, z = f_1D_butter_lowpass_filter(data,cutoff,fs,order,padlen)
        d_pretrain[col] = z
                    
        data = d_pretest[col].values
        y, z = f_1D_butter_lowpass_filter(data,cutoff,fs,order,padlen)
        d_pretest[col] = z
            
        #Scale each col
        mu[col].loc[0] = d_pretrain[col].mean()
        std[col].loc[0] = d_pretrain[col].std()
        
        # Reasoning https://datascience.stackexchange.com/questions/39932/feature-scaling-both-training-and-test-data
        d_pretrain[col] = (d_pretrain[col] - mu[col].values)/std[col].values
        d_pretest[col] = (d_pretest[col] - mu[col].values)/std[col].values
        
    t2 = time()
    #print('Frame number', i_frame[0], 'preprocessing time is', t2-t1, 'seconds')
    return d_pretrain, d_pretest, d_uncertain, mu, std 


def model(df, i_frame, d_hyper):
    t1 = time()
    #print('Fitting model for frame number', i_frame[0], 'out of approximately', i_frame[1], 'total frames')
    lags = d_hyper.h_model[0]
    h_endog = [k for k in d_hyper.h_endog if k]
    
    # Simplistic way to address all NaN values. This should be changed later.
    df = df.asfreq(freq='1H') # this needs to be first as it creates NaN when there is gaps in 1H freq due to time change
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    
    #df = df.diff()[1:]
    #model = ARMA(df, order=(1,0))
    #model = AutoReg(df,lags=[1], old_names=True)
    
    #Split endog and exog
    endog = df[h_endog]
    exog = df.drop(h_endog, axis=1)
    #exog = lagmat(exog, maxlag=1, trim='forward', original='in',use_pandas=True)
    if d_hyper.h_exog.any() == False and len(h_endog) == 1:
        #model = AutoReg(endog,lags=lags, old_names=True)    
        model = sm.tsa.statespace.SARIMAX(endog, order = (3,0,3))
    elif len(h_endog) > 1:
        model = VARMAX(endog, order=lags, exog=exog)
    else:
        #model = AutoReg(endog,lags=lags,exog=exog, old_names=True)
        model = sm.tsa.statespace.SARIMAX(endog, exog, order = (3,0,3))
    
    model_fit = model.fit()

    t2 = time()
    # print('Frame number', i_frame[0], 'model fitting time is', t2-t1, 'seconds')
    return model_fit, exog


def forecast(df1, df2, exog, mu, std, i_frame, d_hyper):
    #df1 is d_test and df2 is d_model = model.fit()
    t1 = time()
    h_framestudy = d_hyper.h_framestudy
    h_endog = [k for k in d_hyper.h_endog if k]

    endog_col = df1.columns.difference(exog.columns)
    d_fore = pd.DataFrame(df1[endog_col], columns=endog_col)

    exog_oos = df1[df1.drop(h_endog, axis=1).columns].asfreq(freq='1H')
    #exog_oos = lagmat(exog_oos, maxlag=1, trim='forward', original='in',use_pandas=True)
    exog_oos = exog_oos.resample('H').first().fillna(method='ffill')
    exog_oos = exog_oos.fillna(method='bfill')

    pred_start_date = exog_oos.index[0]
    pred_end_date = exog_oos.index[-1]
    td = pred_end_date - pred_start_date

    if d_hyper.h_exog.any() == False and len(h_endog) == 1:
        predictions = df2.predict(start = pred_start_date, end = pred_end_date)
        predictions = predictions.to_frame()
    elif len(h_endog) > 1:
        predictions = df2.predict(start = pred_start_date, end = pred_end_date, exog = exog_oos)
    else:
        #predictions = df2.predict(start = pred_start_date, end = pred_end_date,
        #                         exog=exog, exog_oos = exog_oos) # this is for AutoReg
        predictions = df2.predict(start = pred_start_date, end = pred_end_date, exog = exog_oos)
        predictions = predictions.to_frame()

    d_test_unscale = df1
    mu_y = mu[h_endog]
    std_y = std[h_endog]

    d_test_unscale = d_test_unscale*std.values+mu.values
    d_predict_unscale = predictions*std_y.values[0]+mu_y.values[0]
    d_test_unscale = d_test_unscale.iloc[0:h_framestudy[0]]

    #for col in endog_col:
    #    d_fore[col].loc[df1.index[0]:df1.index[-1]] = d_predict_unscale[col]
    d_fore.loc[df1.index[0]:df1.index[-1]] = d_predict_unscale
    d_forec = d_fore.iloc[0:h_framestudy[0]]
    t2 = time()
    #print('Frame number', i_frame[0], 'forecasting time is', t2-t1, 'seconds')
    return d_forec, d_test_unscale

def save_experiment(fileName,d_for,d_tst,d_trials,d_hyper,d_mod,Exp_t,d_raw):
    d_wRMSE = {} # Legacy code, just KST_60
    d_RMSE = {} # Legacy code, just KST_60 
    d_aRMSE = {} # All endog RMSE
    total_length = 0
    for i in range(len(d_for)):
        total_length = total_length + len(d_for[i])
        d_RMSE[i] = np.sqrt(np.mean((d_for[i].KST_60 - d_tst[i].KST_60)**2))
        d_wRMSE[i] = len(d_for[i].KST_60)*np.sqrt(np.mean((d_for[i].KST_60 - d_tst[i].KST_60)**2))
        d_aRMSE[i] = np.sqrt(np.mean((d_for[i] - d_tst[i][d_for[i].columns.values.tolist()])**2)).values.tolist()

    for i in range(len(d_wRMSE)):
        d_wRMSE[i] = d_wRMSE[i]/total_length

    d_nwRMSE = {k: v / total for total in (sum(d_wRMSE.values()),) for k,v in d_wRMSE.items()}
    d_aRMSE = pd.DataFrame.from_dict(d_aRMSE, orient='index',columns=d_for[0].columns.values.tolist())
    d_aRMSE['RMSE-Avg'] = d_aRMSE.mean(axis=1)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(3,1,figsize=(60,20)) # Create matplotlib figure
    fig.subplots_adjust(hspace=.27)
    ax2 = ax[0].twinx() # Create another axes that shares the same x-axis as ax.
    width = 0.4
    x_pos = np.arange(len(d_RMSE))
    b1 = ax[0].bar(x_pos+0, d_RMSE.values(), color='black', width=width, label='RMSE = %.1f' %(round(sum(d_RMSE.values()),2)))
    ax[0].set_ylabel('RMSE', fontsize=20)
    b2 = ax2.bar(x_pos+0.4, d_nwRMSE.values(), color='blue', width=width, label='NW-RMSE = %.1f' %(round(sum(d_nwRMSE.values()),2)))
    ax2.set_ylabel('NW-RMSE',  fontsize=20)
    ax2.grid(None)
    ax[0].set_xlabel('Trial number',  fontsize=20)
    ax[0].legend(handles=[b1,b2], fontsize=14)
    min_trial = min(d_RMSE, key=d_RMSE.get)
    max_trial = max(d_RMSE, key=d_RMSE.get)
    p11, = ax[1].plot(d_trials[min_trial].KST_60.loc[d_tst[min_trial].index[0]:d_tst[min_trial].index[-1]], label='Raw data, trial %.0f' %(min_trial))
    p12, = ax[1].plot(d_tst[min_trial].KST_60, label='Test data')
    p13, = ax[1].plot(d_for[min_trial].KST_60, label='Forecast RMSE = %.1f' %(round(d_RMSE[min_trial],1)))
    ax[1].set_xlabel('Date',fontsize=20)
    ax[1].set_ylabel('60 ft KST ($^\circ$C)', fontsize=20)
    ax[1].legend(handles=[p11,p12,p13], fontsize=14)
    p21, = ax[2].plot(d_trials[max_trial].KST_60.loc[d_tst[max_trial].index[0]:d_tst[max_trial].index[-1]], label='Raw data, trial %.0f' %(max_trial))
    p22, = ax[2].plot(d_tst[max_trial].KST_60, label='Test data')
    p23, = ax[2].plot(d_for[max_trial].KST_60, label='Forecast RMSE = %.1f' %(round(d_RMSE[max_trial],1)))
    ax[2].set_xlabel('Date',fontsize=20)
    ax[2].set_ylabel('60 ft KST ($^\circ$C)', fontsize=20)
    ax[2].legend(handles=[p21,p22,p23], fontsize=14)

    imgdata = BytesIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)  # rewind the data

    drawing=svg2rlg(imgdata)

    d_modsum = pd.Series(index = d_raw.index, name='model')
    for i in range(len(d_mod)):
        for key in d_mod[i]:
            d_modsum[key] = d_mod[i][key]
    d_modsum = d_modsum.dropna()
    d_modsum = d_modsum.to_frame()
    d_pval = pd.DataFrame(index=d_modsum.index, columns = d_modsum['model'][0].params.index.to_list())
    d_sumstat = pd.DataFrame(index=d_modsum.index, columns = ['AIC','BIC','MAE'])
    for i in range(len(d_modsum)):
        d_pval.iloc[i] = d_modsum['model'][i].pvalues.values
        d_sumstat['AIC'].iloc[i] = d_modsum['model'][i].aic
        d_sumstat['BIC'].iloc[i] = d_modsum['model'][i].bic
        #d_sumstat['MAE'].iloc[i] = d_modsum['model'][i].mae

    sumstat_merge = [(d_sumstat.mean().round(4).values[i],
                 d_sumstat.std().round(4).values[i]) for i in range(0,len(d_sumstat.std().values))]
    sumstat_merge = pd.DataFrame(sumstat_merge,index=d_sumstat.columns,columns=['mean','std'])

    labels = ['Cleaning','Startup','Benchmark','Trial','Moving window','Endog variable','Exog variable','Preprocess',
            'Model']
    idx = [['Min-time','Diff limit','Std-dev limit','Mean limit'],[],[],[],['Step-size','Min window','Max window','Horizon'],
            [],[],['Filter order','Sampling frequency','Cutoff frequency','Max KST diff','Min KST value'],['Lag'],
            d_modsum['model'][0].params.index]

    data = [['Total RMSE is ' + str(round(sum(d_aRMSE['RMSE-Avg'].values),2)) + ' in ' + str(Exp_t) + ' seconds'],
        [pd.Series(d_hyper.h_clean.dropna().values,index=idx[0]).to_frame(labels[0])],
        [pd.Series(d_hyper.h_framestudy.dropna().values,index=idx[4]).to_frame(labels[4])],
        [d_hyper.h_endog.dropna()], [d_hyper.h_exog.dropna()],
        [pd.Series(d_hyper.h_preproc.dropna().values,index=idx[7]).to_frame(labels[7])],
        [pd.Series(d_hyper.h_model.dropna().values,index=idx[8]).to_frame(labels[8])],
        [sumstat_merge],[d_aRMSE.round(3)]]

    pval_merge = [(d_pval.mean().round(6).values[i],
                 d_pval.std().round(6).values[i]) for i in range(0,len(d_pval.mean().values))]
    
    data2 = pd.DataFrame(pval_merge,index=idx[9],columns=['mean','std'])
    data2 = data2.reset_index().values.tolist()
    data2 = [['pvalues', 'mean', 'std-dev']]+data2
    
    table = Table(data)
    table2 = Table(data2)

    # add style
    style = TableStyle([
        ('BACKGROUND', (0,0), (3,0), colors.green),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 16),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('LEADING', (0,0), (-1,-1), 16)
    ])

    table.setStyle(style)
    table2.setStyle(style)

    # 2) Alternate background color
    rowNumb = len(data)
    for i in range(1,rowNumb):
        if i % 2 == 0:
            bc = colors.burlywood
        else:
            bc = colors.beige

        ts = TableStyle(
            [('BACKGROUND', (0,i), (-1,i), bc),
            ]
            )
        table.setStyle(ts)
        table2.setStyle(ts)

    # 3) Add borders
    ts = TableStyle(
        [
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('LINEBEFORE',(2,1),(2,-1),2,colors.red),
        ('LINEABOVE',(0,2),(-1,2),2,colors.green),
        ('GRID',(0,1),(-1,-1),2,colors.black)
        ]
    )
    table.setStyle(ts)
    table2.setStyle(ts)

    elems = []
    elems2= []
    elems.append(table)
    elems2.append(table2)
    #   pdf.build(elems)

    c = canvas.Canvas(fileName)
    c.setPageSize((5400, 1600))
    renderPDF.draw(drawing,c, 400, -30)
    w, h = table.wrapOn(c, 0, 0)
    table.drawOn(c,10,10)
    w2, h2 = table2.wrapOn(c, 0, 0)
    table2.drawOn(c,350,10)
    c.showPage()
    c.save()




# This function is preamble for the 5yrx5min data for residual carbonate prediction

def Five_Min():
    # Load and format data
    path1 = 'D:\Canfor_Data_Confidential\Data_From_Cilius\Fiveyr5minData1\Fiveyr5minData1.csv'
    df1 = pd.read_csv(path1)

    tags = df1.iloc[0:3,1:]
    tags['Variable'] = ['Tag','Unit','Short']
    tags = tags.set_index('Variable')
    tags.loc[4] = tags.columns.values.tolist()
    tags.columns = tags.loc['Short'].values.tolist()
    tags.loc['Short'] = tags.loc['Unit'].values.tolist()
    tags.loc['Unit'] = tags.loc['Tag'].values.tolist()
    tags.loc['Tag'] = tags.loc[4].values.tolist()
    tags = tags.drop(4)
    tags.iloc[2,60:67] = ['PB_F1_MW', 'PB_F2_MW', 'SCRB_M_D','PB_REC','FRE_D','FDE_D','K_D']

    df1  = df1.set_index(df1.iloc[:,0])
    df1.index.names= ['datetime']
    df1 = df1.drop(df1.columns[[0]],axis=1)
    df1 = df1.drop(df1.index[[0,1,2]])
    df1.index = pd.to_datetime(df1.index)
    old_cols1 = df1.columns
    df1.rename(columns=dict(zip(old_cols1, tags.loc['Short'].values)), inplace = True)
    data_columns = df1.columns
    df1 = (df1.drop(data_columns, axis=1).join(df1[data_columns].apply(pd.to_numeric, errors='coerce')))

    path2 = 'D:\Canfor_Data_Confidential\Data_From_Cilius\Fiveyr5minData2\Fiveyr5minData2.csv'
    df2 = pd.read_csv(path2)

    df2  = df2.set_index(df2.iloc[:,0])
    df2.index.names= ['datetime']
    df2 = df2.drop(df2.columns[[0]],axis=1)
    df2 = df2.drop(df2.index[[0,1,2]])
    df2.index = pd.to_datetime(df2.index)
    old_cols2 = df2.columns
    df2.rename(columns=dict(zip(old_cols2, tags.loc['Short'].values)), inplace = True)
    data_columns = df2.columns
    df2 = (df2.drop(data_columns, axis=1).join(df2[data_columns].apply(pd.to_numeric, errors='coerce')))

    path3 = 'D:\Canfor_Data_Confidential\Data_From_Cilius\Fiveyr5minData3\Fiveyr5minData3.csv'
    df3 = pd.read_csv(path3)

    df3  = df3.set_index(df3.iloc[:,0])
    df3.index.names= ['datetime']
    df3 = df3.drop(df3.columns[[0]],axis=1)
    df3 = df3.drop(df3.index[[0,1,2]])
    df3.index = pd.to_datetime(df3.index)
    old_cols3 = df3.columns
    df3.rename(columns=dict(zip(old_cols3, tags.loc['Short'].values)), inplace = True)
    data_columns = df3.columns
    df3 = (df3.drop(data_columns, axis=1).join(df3[data_columns].apply(pd.to_numeric, errors='coerce')))

    path4 = 'D:\Canfor_Data_Confidential\Data_From_Cilius\Fiveyr5minData4\Fiveyr5minData4.csv'
    df4 = pd.read_csv(path4)

    df4  = df4.set_index(df4.iloc[:,0])
    df4.index.names= ['datetime']
    df4 = df4.drop(df4.columns[[0]],axis=1)
    df4 = df4.drop(df4.index[[0,1,2]])
    df4.index = pd.to_datetime(df4.index)
    old_cols4 = df4.columns
    df4.rename(columns=dict(zip(old_cols4, tags.loc['Short'].values)), inplace = True)
    data_columns = df4.columns
    df4 = (df4.drop(data_columns, axis=1).join(df4[data_columns].apply(pd.to_numeric, errors='coerce')))

    df = df1.append(df2)
    df = df.append(df3)
    df = df.append(df4)

    # Add Primary Air data to DF
    path8 = 'D:\Canfor_Data_Confidential\Data_From_Cilius\Fiveyr5minData_primaryAir\Fiveyr5minData_PA.csv'
    df0 = pd.read_csv(path8,encoding = "ISO-8859-1")
    tags[df0.iloc[2,1]] = [df0.columns[1], df0.iloc[0,1], df0.iloc[1,1]]
    df0  = df0.set_index(df0.iloc[:,0])
    df0.index.names= ['datetime']
    df0 = df0.drop(df0.columns[[0]],axis=1)
    df0 = df0.iloc[3:]
    df0.index = pd.to_datetime(df0.index)
    data_columns = df0.columns
    df0 = (df0.drop(data_columns, axis=1).join(df0[data_columns].apply(pd.to_numeric, errors='coerce')))
    df0 = df0[~df0.index.duplicated(keep='first')] # Remove duplicate indices from Daylight Savings Time
    df0 = df0.resample('5min').pad()
    df0 = df0.replace([np.inf, -np.inf], np.nan)
    df0 = df0.astype('float')
    df[tags.iloc[-1,-1]] = df0

    df = df[~df.index.duplicated(keep='first')] # Remove duplicate indices from Daylight Savings Time
    df = df.resample('5min').pad()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype('float')

    # Outlier removal based on values
    # In df.where, when the condition == True, the value is used.
    df['1_FAS'].where((df['1_FAS'] <= 100) & (df['1_FAS'] >= 0), inplace=True)
    df['M21F'].where((df['M21F'] <= 500) & (df['M21F'] >= 0), inplace=True)
    df['1_MFV'].where((df['1_MFV'] <= 30) & (df['1_MFV'] >= -20), inplace=True)
    df['2_FAS'].where((df['2_FAS'] <= 100) & (df['2_FAS'] >= 0), inplace=True)
    df['M22F'].where((df['M22F'] <= 500) & (df['M22F'] >= 0), inplace=True)
    df['2_MFV'].where((df['2_MFV'] <= 30) & (df['2_MFV'] >= -20), inplace=True)
    df['KS_MF_T'].where((df['KS_MF_T'] <= 800) & (df['KS_MF_T'] >= 0), inplace=True)
    df['KF'].where((df['KF'] <= 800) & (df['KF'] >= 0), inplace=True)
    df['FDE_T'].where((df['FDE_T'] <= 400) & (df['FDE_T'] >= 0), inplace=True)
    df['K_FDE_T'].where((df['K_FDE_T'] <= 400) & (df['K_FDE_T'] >= 0), inplace=True)
    df['ID_FI_T'].where((df['ID_FI_T'] <= 400) & (df['ID_FI_T'] >= 0), inplace=True)
    df['LKS'].where((df['LKS'] <= 4) & (df['LKS'] >= 0), inplace=True)
    df['WLS_T'].where((df['WLS_T'] <= 100) & (df['WLS_T'] >= 0), inplace=True)
    df['K_FDE_O2'].where((df['K_FDE_O2'] <= 100) & (df['K_FDE_O2'] >= 0), inplace=True)
    df['KO'].where((df['KO'] <= 100) & (df['KO'] >= 0), inplace=True) # 15
    df['K_FRE_TR'].where((df['K_FRE_TR'] <= 1300) & (df['K_FRE_TR'] >= 0), inplace=True)
    df['K_FRE_TC'].where((df['K_FRE_TC'] <= 1300) & (df['K_FRE_TC'] >= 0), inplace=True)
    df['KFS'].where((df['KFS'] <= 1300) & (df['KFS'] >= 0), inplace=True)
    df['BMZ_T'].where((df['BMZ_T'] <= 900) & (df['BMZ_T'] >= 0), inplace=True)
    df['FMZ_T'].where((df['FMZ_T'] <= 1000) & (df['FMZ_T'] >= 0), inplace=True)
    df['NG_2K'].where((df['NG_2K'] <= 4000) & (df['NG_2K'] >= 0), inplace=True)
    df['CACO3_T'].where((df['CACO3_T'] <= 100) & (df['CACO3_T'] >= 0), inplace=True)
    df['WL_T_pS'].where((df['WL_T_pS'] <= 100) & (df['WL_T_pS'] >= 0), inplace=True)
    df['LQ_EST'].where((df['LQ_EST'] <= 100) & (df['LQ_EST'] >= 0), inplace=True)
    df['1_CST'].where((df['1_CST'] <= 30) & (df['1_CST'] >= 0), inplace=True)
    df['WL_T_pCE'].where((df['WL_T_pCE'] <= 100) & (df['WL_T_pCE'] >= 0), inplace=True)
    df['C1_pCE'].where((df['C1_pCE'] <= 100) & (df['C1_pCE'] >= 0), inplace=True)
    df['W_S'].where((df['W_S'] <= 70) & (df['W_S'] >= 0), inplace=True)
    df['A_T'].where((df['A_T'] <= 50) & (df['A_T'] >= -50), inplace=True)
    df['SHS_0_90'].where((df['SHS_0_90'] <= 900) & (df['SHS_0_90'] >= 0), inplace=True) # 30
    df['K_LB_S'].where((df['K_LB_S'] <= 600) & (df['K_LB_S'] >= 0), inplace=True)
    df['K_Beam'].where((df['K_Beam'] <= 300) & (df['K_Beam'] >= 0), inplace=True)
    df['Gearbox'].where((df['Gearbox'] <= 300) & (df['Gearbox'] >= 0), inplace=True)
    df['SHS_0_180'].where((df['SHS_0_180'] <= 300) & (df['SHS_0_180'] >= 0), inplace=True)
    df['Dump_Gate'].where((df['Dump_Gate'] <= 300) & (df['Dump_Gate'] >= 0), inplace=True)
    df['WW_TTA'].where((df['WW_TTA'] <= 250) & (df['WW_TTA'] >= 0), inplace=True)
    KST_cols = [col for col in df.columns if 'KST' in col]
    for col in KST_cols:
        df[col].where((df[col] <= 800) & (df[col] >= 0), inplace=True)
    df['PB_F1_MW'].where((df['PB_F1_MW'] <= 25) & (df['PB_F1_MW'] >= 0), inplace=True)
    df['PB_F2_MW'].where((df['PB_F2_MW'] <= 25) & (df['PB_F2_MW'] >= 0), inplace=True)
    df['SCRB_M_D'].where((df['SCRB_M_D'] <= 10) & (df['SCRB_M_D'] >= 0), inplace=True)
    df['PB_REC'].where((df['PB_REC'] <= 100) & (df['PB_REC'] >= 0), inplace=True)
    df['FRE_D'].where((df['FRE_D'] <= 5) & (df['FRE_D'] >= -5), inplace=True)
    df['FDE_D'].where((df['FDE_D'] <= 10) & (df['FDE_D'] >= -10), inplace=True)
    df['K_D'].where((df['K_D'] <= 100) & (df['K_D'] >= 0), inplace=True)


    # Outlier removal based on change of values
    df['1_FAS'].where((df['1_FAS'].diff() <= 25) & (df['1_FAS'].diff() >= -25), inplace=True)
    df['M21F'].where((df['M21F'].diff() <= 300) & (df['M21F'].diff() >= -300), inplace=True)
    df['1_MFV'].where((df['1_MFV'].diff() <= 20) & (df['1_MFV'].diff() >= -20), inplace=True)
    df['2_FAS'].where((df['2_FAS'].diff() <= 25) & (df['2_FAS'].diff() >= -25), inplace=True)
    df['M22F'].where((df['M22F'].diff() <=300) & (df['M22F'].diff() >= -300), inplace=True)
    df['2_MFV'].where((df['2_MFV'].diff() <= 20) & (df['2_MFV'].diff() >= -20), inplace=True)
    df['KS_MF_T'].where((df['KS_MF_T'].diff() <= 400) & (df['KS_MF_T'].diff() >= -400), inplace=True)
    df['KF'].where((df['KF'].diff() <= 400) & (df['KF'].diff() >= -400), inplace=True)
    df['FDE_T'].where((df['FDE_T'].diff() <= 50) & (df['FDE_T'].diff() >= -50), inplace=True)
    df['K_FDE_T'].where((df['K_FDE_T'].diff() <= 50) & (df['K_FDE_T'].diff() >= -50), inplace=True)
    df['ID_FI_T'].where((df['ID_FI_T'].diff() <= 50) & (df['ID_FI_T'].diff() >= -50), inplace=True)
    df['LKS'].where((df['LKS'].diff() <= 1) & (df['LKS'].diff() >= -1), inplace=True)
    df['WLS_T'].where((df['WLS_T'].diff() <= 20) & (df['WLS_T'].diff() >= -20), inplace=True)
    df['K_FDE_O2'].where((df['K_FDE_O2'].diff() <= 10) & (df['K_FDE_O2'].diff() >= -10), inplace=True)
    df['KO'].where((df['KO'].diff() <= 10) & (df['KO'].diff() >= -10), inplace=True) # 15
    df['K_FRE_TR'].where((df['K_FRE_TR'].diff() <= 300) & (df['K_FRE_TR'].diff() >= -300), inplace=True)
    df['K_FRE_TC'].where((df['K_FRE_TC'].diff() <= 300) & (df['K_FRE_TC'].diff() >= -300), inplace=True)
    df['KFS'].where((df['KFS'].diff() <= 500) & (df['KFS'].diff() >= -500), inplace=True)
    df['BMZ_T'].where((df['BMZ_T'].diff() <= 100) & (df['BMZ_T'].diff() >= -100), inplace=True)
    df['FMZ_T'].where((df['FMZ_T'].diff() <= 100) & (df['FMZ_T'].diff() >= -100), inplace=True)
    df['NG_2K'].where((df['NG_2K'].diff() <= 500) & (df['NG_2K'].diff() >= -500), inplace=True)
    df['CACO3_T'].where((df['CACO3_T'].diff() <= 20) & (df['CACO3_T'].diff() >= -20), inplace=True)
    df['WL_T_pS'].where((df['WL_T_pS'].diff() <= 10) & (df['WL_T_pS'].diff() >= -10), inplace=True)
    df['LQ_EST'].where((df['LQ_EST'].diff() <= 20) & (df['LQ_EST'].diff() >= -20), inplace=True)
    df['1_CST'].where((df['1_CST'].diff() <= 10) & (df['1_CST'].diff() >= -10), inplace=True)
    df['WL_T_pCE'].where((df['WL_T_pCE'].diff() <= 20) & (df['WL_T_pCE'].diff() >= -20), inplace=True)
    df['C1_pCE'].where((df['C1_pCE'].diff() <= 20) & (df['C1_pCE'].diff() >= -20), inplace=True)
    df['W_S'].where((df['W_S'].diff() <= 20) & (df['W_S'].diff() >= -20), inplace=True)
    df['A_T'].where((df['A_T'].diff() <= 5) & (df['A_T'].diff() >= -5), inplace=True)
    df['SHS_0_90'].where((df['SHS_0_90'].diff() <= 200) & (df['SHS_0_90'].diff() >= -200), inplace=True) # 30
    df['K_LB_S'].where((df['K_LB_S'].diff() <= 50) & (df['K_LB_S'].diff() >= -50), inplace=True)
    df['K_Beam'].where((df['K_Beam'].diff() <= 50) & (df['K_Beam'].diff() >= -50), inplace=True)
    df['Gearbox'].where((df['Gearbox'].diff() <= 50) & (df['Gearbox'].diff() >= -50), inplace=True)
    df['SHS_0_180'].where((df['SHS_0_180'].diff() <= 100) & (df['SHS_0_180'].diff() >= -100), inplace=True)
    df['Dump_Gate'].where((df['Dump_Gate'].diff() <= 50) & (df['Dump_Gate'].diff() >= -50), inplace=True)
    df['WW_TTA'].where((df['WW_TTA'].diff() <= 30) & (df['WW_TTA'].diff() >= -30), inplace=True)

    for col in KST_cols:
        df[col].where((df[col].diff() <= 50) & (df[col].diff() >= -50), inplace=True)

    df['PB_F1_MW'].where((df['PB_F1_MW'].diff() <= 15) & (df['PB_F1_MW'].diff() >= -15), inplace=True)
    df['PB_F2_MW'].where((df['PB_F2_MW'].diff() <= 15) & (df['PB_F2_MW'].diff() >= -15), inplace=True)
    df['SCRB_M_D'].where((df['SCRB_M_D'].diff() <= 0.5) & (df['SCRB_M_D'].diff() >= -0.5), inplace=True)
    df['PB_REC'].where((df['PB_REC'].diff() <= 10) & (df['PB_REC'].diff() >= -10), inplace=True)
    df['FRE_D'].where((df['FRE_D'].diff() <= 1) & (df['FRE_D'].diff() >= -1), inplace=True)
    df['FDE_D'].where((df['FDE_D'].diff() <= 2) & (df['FDE_D'].diff() >= -2), inplace=True)
    df['K_D'].where((df['K_D'].diff() <= 50) & (df['K_D'].diff() >= -50), inplace=True)

    return df, tags







# The two functions below were merged into the save_experiment function above.
'''
def result_summary(d_for,d_tst,d_trials,d_hyper):
    t1 = time()
    d_wRMSE = {}
    d_RMSE = {}
    total_length = 0
    for i in range(len(d_for)):
        total_length = total_length + len(d_for[i])
        d_RMSE[i] = np.sqrt(np.mean((d_for[i].KST_60 - d_tst[i].KST_60)**2))
        d_wRMSE[i] = len(d_for[i].KST_60)*np.sqrt(np.mean((d_for[i].KST_60 - d_tst[i].KST_60)**2))

    for i in range(len(d_wRMSE)):
        d_wRMSE[i] = d_wRMSE[i]/total_length

    d_nwRMSE = {k: v / total for total in (sum(d_wRMSE.values()),) for k,v in d_wRMSE.items()}

    plt.style.use('ggplot')
    fig, ax = plt.subplots(3,1,figsize=(20,12)) # Create matplotlib figure
    fig.subplots_adjust(hspace=.27)
    ax2 = ax[0].twinx() # Create another axes that shares the same x-axis as ax.
    width = 0.4
    x_pos = np.arange(len(d_RMSE))
    b1 = ax[0].bar(x_pos+0, d_RMSE.values(), color='black', width=width, label='RMSE = %.1f' %(round(sum(d_RMSE.values()),2)))
    ax[0].set_ylabel('RMSE', fontsize=20)
    b2 = ax2.bar(x_pos+0.4, d_nwRMSE.values(), color='blue', width=width, label='NW-RMSE = %.1f' %(round(sum(d_nwRMSE.values()),2)))
    ax2.set_ylabel('NW-RMSE',  fontsize=20)
    ax2.grid(None)
    ax[0].set_xlabel('Trial number',  fontsize=20)
    ax[0].legend(handles=[b1,b2], fontsize=14)
    min_trial = min(d_RMSE, key=d_RMSE.get)
    max_trial = max(d_RMSE, key=d_RMSE.get)
    p11, = ax[1].plot(d_trials[min_trial].KST_60.loc[d_tst[min_trial].index[0]:d_tst[min_trial].index[-1]], label='Raw data, trial %.0f' %(min_trial))
    p12, = ax[1].plot(d_tst[min_trial].KST_60, label='Test data')
    p13, = ax[1].plot(d_for[min_trial].KST_60, label='Forecast RMSE = %.1f' %(round(d_RMSE[min_trial],1)))
    ax[1].set_xlabel('Date',fontsize=20)
    ax[1].set_ylabel(d_hyper.h_endog[0]+' ($^\circ$C)', fontsize=20)
    ax[1].legend(handles=[p11,p12,p13], fontsize=14)
    p21, = ax[2].plot(d_trials[max_trial].KST_60.loc[d_tst[max_trial].index[0]:d_tst[max_trial].index[-1]], label='Raw data, trial %.0f' %(max_trial))
    p22, = ax[2].plot(d_tst[max_trial].KST_60, label='Test data')
    p23, = ax[2].plot(d_for[max_trial].KST_60, label='Forecast RMSE = %.1f' %(round(d_RMSE[max_trial],1)))
    ax[2].set_xlabel('Date',fontsize=20)
    ax[2].set_ylabel(d_hyper.h_endog[0]+' ($^\circ$C)', fontsize=20)
    ax[2].legend(handles=[p21,p22,p23], fontsize=14)
    plt.show()

    t2 = time()
    print('Summarizing results in', t2-t1, 'seconds')
    return d_RMSE

def save_experiment(fileName,d_RMSE, d_hyper, d_modelsummary, Exp_t):
    t1 = time()
    # Remove NaN values from big model summary dataframe
    d_model = d_modelsummary.dropna()
    # Extract p-values into a dataframe to average them
    d_pval = pd.DataFrame(d_model.pvalues.tolist(),index=d_model.index, columns = d_model.params[0])

    labels = ['Cleaning','Startup','Benchmark','Trial','Moving window','Endog variable','Exog variable','Preprocess',
            'Model','p-values','RMSE of Trials']
    idx = [['Diff limit','Std-dev limit','Mean limit'],[],[],[],['Step-size','Min window','Max window','Horizon'],
            [],[],['Filter order','Sampling frequency','Cutoff frequency','Max KST diff','Min KST value'],['Lag'],
            d_model.params[0]]

    data = [['Total RMSE is ' + str(round(sum(d_RMSE.values()),2)) + ' in ' + str(Exp_t) + ' seconds'],
        [pd.Series(d_hyper.h_clean.dropna().values,index=idx[0]).to_frame(labels[0])],
        [pd.Series(d_hyper.h_framestudy.dropna().values,index=idx[4]).to_frame(labels[4])],
        [d_hyper.h_endog.dropna()], [d_hyper.h_exog.dropna()],
        [pd.Series(d_hyper.h_preproc.dropna().values,index=idx[7]).to_frame(labels[7])],
        [pd.Series(d_hyper.h_model.dropna().values,index=idx[8]).to_frame(labels[8])],
        [pd.Series(d_pval.mean().values, index=idx[9]).to_frame(labels[9])],
        [pd.Series(d_RMSE).to_frame(labels[10])]]

    pdf = SimpleDocTemplate(
        fileName,
        pagesize=letter
    )

    table = Table(data)

    # add style
    style = TableStyle([
        ('BACKGROUND', (0,0), (3,0), colors.green),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige)
    ])

    table.setStyle(style)

    # 2) Alternate background color
    rowNumb = len(data)
    for i in range(1,rowNumb):
        if i % 2 == 0:
            bc = colors.burlywood
        else:
            bc = colors.beige

        ts = TableStyle(
            [('BACKGROUND', (0,i), (-1,i), bc)]
            )
        table.setStyle(ts)


    # 3) Add borders
    ts = TableStyle(
        [
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('LINEBEFORE',(2,1),(2,-1),2,colors.red),
        ('LINEABOVE',(0,2),(-1,2),2,colors.green),
        ('GRID',(0,1),(-1,-1),2,colors.black)
        ]
    )
    table.setStyle(ts)

    elems = []
    elems.append(table)

    pdf.build(elems)

    t2 = time()
    print('Storing table in', t2-t1, 'seconds')
'''

# from statsmodels.tsa.ar_model import ar_select_order
# mod = ar_select_order(d_tst[13].KST_60,maxlag=60)