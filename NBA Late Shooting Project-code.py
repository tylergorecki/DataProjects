#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import unittest

#%%

# read in data
nba_data = pd.read_csv("/Users/tylergorecki/Desktop/DS 2001/2019-20_pbp.csv")

# clean data
def clean_shots(nba_data2):
    shots = nba_data2.loc[nba_data2.loc[:,"ShotOutcome"].isna() == False]
    shots.loc[:,'time'] = np.where(((shots.loc[:,'Quarter'] > 3) & (shots.loc[:,'SecLeft'] <= 60)), 'late', 'notLate')
    # print(shots['SecLeft'])
    mod_shots1 = shots.loc[:,['ShotType', 'ShotOutcome', 'ShotDist', 'time']]
    # print(mod_shots)
    return mod_shots1, shots

mod_shots, shots = clean_shots(nba_data)

# saved to csv
mod_shots.to_csv('/Users/tylergorecki/Desktop/DS 2001/lateShotsNBA.csv')

#%%

# creating separate datasets for each time value
notLate = mod_shots.loc[mod_shots.time=='notLate']
late = mod_shots.loc[mod_shots.time=='late']

# Proportion of makes and misses for each time value
notLate.loc[:,'ShotOutcome'].value_counts()/len(notLate)
late.loc[:,'ShotOutcome'].value_counts()/len(late)

# testing distances distribution
distances = pd.Series(np.random.gamma(10, size = 1000)*1.5)
distances.plot.hist(grid=True, bins = 5, rwidth = 0.9, color = 'red')

# testing other visualizations (not final)
distances = plt.hist(mod_shots.loc[:,'ShotType'], rwidth = 0.9)
distances = plt.hist(notLate.loc[:,'ShotDist'], range=(0,40))
distances2 = plt.hist(late.loc[:,'ShotDist'], range=(0,40))

#%%

# VISUALIZATION 1: proportion of makes at each distance for each time

X = ['2-pt dunk','2-pt layup','2-pt hook shot','2-pt jump shot','3-pt jump shot']
  
X_axis = np.arange(len(X))*2

twoD = notLate.loc[notLate.loc[:,'ShotType']=='2-pt dunk']
twoD_nL = len(twoD.loc[notLate.loc[:,'ShotOutcome']=='make'])/len(twoD)

twoL = notLate.loc[notLate.loc[:,'ShotType']=='2-pt layup']
twoL_nL = len(twoL.loc[notLate.loc[:,'ShotOutcome']=='make'])/len(twoL)

twoH = notLate.loc[notLate.loc[:,'ShotType']=='2-pt hook shot']
twoH_nL = len(twoH.loc[notLate.loc[:,'ShotOutcome']=='make'])/len(twoH)

twoJ = notLate.loc[notLate.loc[:,'ShotType']=='2-pt jump shot']
twoJ_nL = len(twoJ.loc[notLate.loc[:,'ShotOutcome']=='make'])/len(twoJ)

threeJ = notLate.loc[notLate.loc[:,'ShotType']=='3-pt jump shot']
threeJ_nL = len(threeJ.loc[notLate.loc[:,'ShotOutcome']=='make'])/len(threeJ)

twoDlate = late.loc[late.loc[:,'ShotType']=='2-pt dunk']
twoD_l = len(twoDlate.loc[late.loc[:,'ShotOutcome']=='make'])/len(twoDlate)

twoLlate = late.loc[late.loc[:,'ShotType']=='2-pt layup']
twoL_l = len(twoLlate.loc[late.loc[:,'ShotOutcome']=='make'])/len(twoLlate)

twoHlate = late.loc[late.loc[:,'ShotType']=='2-pt hook shot']
twoH_l = len(twoHlate.loc[late.loc[:,'ShotOutcome']=='make'])/len(twoHlate)

twoJlate = late.loc[late.loc[:,'ShotType']=='2-pt jump shot']
twoJ_l = len(twoJlate.loc[late.loc[:,'ShotOutcome']=='make'])/len(twoJlate)

threeJlate = late.loc[late.loc[:,'ShotType']=='3-pt jump shot']
threeJ_l = len(threeJlate.loc[late.loc[:,'ShotOutcome']=='make'])/len(threeJlate)

nL = [twoD_nL, twoL_nL, twoH_nL, twoJ_nL, threeJ_nL]

l = [twoD_l, twoL_l, twoH_l, twoJ_l, threeJ_l]
  
plt.bar(X_axis - 0.4, nL, .8, label = 'Not Late')
plt.bar(X_axis + 0.4, l, .8, label = 'Late')
  
plt.xticks(X_axis, X, rotation=15)
plt.minorticks_on()
plt.xlabel("Shot types")
plt.ylabel("Proportion of makes")
plt.title("Shooting percentage for each shot type by time level")
plt.legend()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.show()

#%%

# VISUALIZATION 2

plt.hist(notLate.ShotDist, weights = np.ones(len(notLate))/len(notLate), range=(0,40), alpha=0.5, bins=40)
plt.gca().set_ylim([0,.13])

plt.hist(late.ShotDist, weights = np.ones(len(late))/len(late), range=(0,40), alpha=0.5, bins=40)
plt.gca().set_ylim([0,.13])

plt.minorticks_on()
plt.xlabel("Shot distances")
plt.ylabel("Proportion of shots at each time level")
plt.title("Proportion of shots taken at each distance by time level")
plt.legend(['Not late','Late'])
plt.show()

#%%

# two sample proportion test
success_notLate = len(notLate[notLate.ShotOutcome=='make'])
size_notLate = len(notLate)
success_late = len(late[late.ShotOutcome=='make'])
size_late = len(late)

successes = np.array([success_notLate, success_late])
samples = np.array([size_notLate, size_late])

# null, H0: p1 = p2
# alternative, Ha: p1 != p2
stat1, p_value1 = proportions_ztest(count=successes, nobs=samples,  alternative='two-sided')
p_value1 
# 6.874*e-11
# there is enough evidence to suggest that the proportion of successes in the 
# two samples is not equal

# null, H0: p1 = p2
# alternative, Ha: p1 > p2
stat, p_value = proportions_ztest(count=successes, nobs=samples,  alternative='larger')
p_value # 3.437*e-11
# there is enough evidence to suggest that the proportion of successes in the 
# sample of not late shots is greater than the proportion of successes in the 
# sample of late shots

# proportion test function (to be unit tested)
def prop_test(a, b):
    success_notLate = len(notLate[notLate.ShotOutcome == 'make'])
    size_notLate = len(notLate)
    success_late = len(late[late.ShotOutcome == 'make'])
    size_late = len(late)

    successes = np.array([success_notLate, success_late])
    samples = np.array([size_notLate, size_late])

    stat1, p_value1 = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')
    stat2, p_value2 = proportions_ztest(count=successes, nobs=samples, alternative='larger')
    return p_value1, p_value2

#%%

notLateDunks = notLate.loc[notLate.loc[:,'ShotType']=='2-pt dunk']
notLateLayups = notLate.loc[notLate.loc[:,'ShotType']=='2-pt layup']
notLateHookshots = notLate.loc[notLate.loc[:,'ShotType']=='2-pt hook shot']
notLate2Jumps = notLate.loc[notLate.loc[:,'ShotType']=='2-pt jump shot']
notLate3Jumps = notLate.loc[notLate.loc[:,'ShotType']=='3-pt jump shot']

lateDunks = late.loc[late.loc[:,'ShotType']=='2-pt dunk']
lateLayups = late.loc[late.loc[:,'ShotType']=='2-pt layup']
lateHookshots = late.loc[late.loc[:,'ShotType']=='2-pt hook shot']
late2Jumps = late.loc[late.loc[:,'ShotType']=='2-pt jump shot']
late3Jumps = late.loc[late.loc[:,'ShotType']=='3-pt jump shot']


def prop_test_new(a_notLate, b_late):
    success_notLate = len(a_notLate[a_notLate.ShotOutcome == 'make'])
    size_notLate = len(a_notLate)
    success_late = len(b_late[b_late.ShotOutcome == 'make'])
    size_late = len(b_late)

    successes = np.array([success_notLate, success_late])
    samples = np.array([size_notLate, size_late])

    stat1, p_value1 = proportions_ztest(count=successes, nobs=samples, alternative='smaller')
    stat2, p_value2 = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')
    stat3, p_value3 = proportions_ztest(count=successes, nobs=samples, alternative='larger')
    return p_value1, p_value2, p_value3

prop_test_new(notLateDunks, lateDunks)
# (0.005991840970411957, 0.011983681940823915, 0.9940081590295881)

prop_test_new(notLateLayups, lateLayups)
# (0.258581699084237, 0.517163398168474, 0.741418300915763)

prop_test_new(notLateHookshots, lateHookshots)
# (0.6633015010574019, 0.6733969978851962, 0.3366984989425981)

prop_test_new(notLate2Jumps, late2Jumps)
# (0.9999891651388686, 2.166972226281242e-05, 1.083486113140621e-05)

prop_test_new(notLate3Jumps, late3Jumps)
# (0.9999999999957517, 8.496590498769555e-12, 4.248295249384778e-12)

