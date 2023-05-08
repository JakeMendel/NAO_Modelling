#%%
import functions_and_classes as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field
from typing import Union, Tuple, Iterable, Optional, Any, Type, Callable, Dict
from tqdm import tqdm
from fancy_einsum import einsum
from scipy import special
from itertools import chain

#%%
for i in range(2):
    mpl.style.use('seaborn-v0_8')
    fontsize = 15
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['figure.figsize'] = (20,10)
    plt.plot(range(5))
    plt.title('Testing')

#%%
#Create Plot of Advantage vs Mixing Rate

mixing_rates = np.logspace(0,6,6,base=10)
delta_t = 0.04
epidemic_time = 70
I_0 = 100
n_sims = 1000
figsavename = None
filters = F.create_arrival_municipal_filter()

basic_datasets = {}
for mixing_rate in mixing_rates:
    key = f'Mixing_Rate: {int(mixing_rate)}'
    sim = F.HomogeneousNetwork(F.BasicCity,2,mixnumber=mixing_rate)
    dataset = sim.multiple_sims(delta_t, epidemic_time,F.measles,I_0,n_sims, moving_avg = False)
    daily = dataset.daily_avg()
    basic_datasets[key] = daily
    print(f'Mixing_Rate {int(mixing_rate)} complete')

thresholds = np.array([0.2])
calculation = F.differences_vs_threshold_data(basic_datasets, filters, thresholds)

F.differences_vs_variable_final_img(basic_datasets, list(mixing_rates), calculation,filters=filters,old_thresholds = thresholds, new_thresholds = thresholds, figsavename = figsavename)
#%%
#Create Plot to compare E to I in the first city over time.
data = list(basic_datasets.values())[0]
f1 = F.create_filter([0],['municipal'],compartments=['I'])
f2 = F.create_filter([0],['municipal'],compartments=['E'])
F.plot_infection_ratio(data,[f1,f2], 'times')


#%%
#Create Plot of Advantage vs Threshold Ratio for a range of airplane thresholds

mixnumber = 1000
I_0 = 1
n_sims = 1000
delta_t = 0.04
epidemic_time = 70
n_points = 100
airport_thresholds = np.array([0.2,0.02,0.001])
error_bars = 'std'
filters = F.create_arrival_municipal_filter()

test = F.HomogeneousNetwork(F.BasicCity, 2, mixnumber=mixnumber)
testdata = test.multiple_sims(delta_t, epidemic_time,F.measles,I_0,n_sims)
testdata = testdata.daily_avg()

F.threshold_ratio_diffs(testdata,filters,n_points = n_points, xlims_log = list(np.log10(airport_thresholds)),error_bars=error_bars)

#%%
#Create a plot of Advantage vs Mixing Likelihood Ratio for FrequentFlyerCity
mixing_LRs = [1,2,3,6,10,20,30,60,100,200,300,600,1000]
I_0 = 1
n_sims = 1000
delta_t = 0.04
epidemic_time = 70
thresholds = 10**np.linspace(-3,-1,3)
mixnumber = 3000
p_ff = 0.99
frequent_flier_frac = 0.01

datasets = {}
for mixing_LR in mixing_LRs:
    key = f'Mixing_LR: {mixing_LR}'
    sim = F.HomogeneousNetwork(F.FrequentFlierCity,2,mixnumber=mixnumber,p_ff = p_ff, frequent_flier_frac = frequent_flier_frac, mixing_LR = mixing_LR)
    dataset = sim.multiple_sims(delta_t,epidemic_time,F.measles,I_0, n_sims,moving_avg = False)
    daily = dataset.daily_avg()
    datasets[key] = daily
    print(f'Mixing_LR {mixing_LR} complete')

threshold_data = F.differences_vs_threshold_data(datasets, filters, thresholds)
F.differences_vs_variable(datasets,mixing_LRs,threshold_data,filters,thresholds,thresholds,log= 'x', x_name='Mixing Likelihood Ratio')
# %%
