# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:49:48 2022

@author: uqfegger
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from statsmodels.tsa.stattools import acf
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.seasonal import seasonal_decompose, STL

#import statsmodels.api as sm
#from statsmodels.tsa.ar_model import AutoReg

# consider a signal which has 12.5 hour, monthly and annual periods
# e.g. a tide

# define the three periods in days
Tp_1 = 0.5175
Tp_2 = 28
Tp_3 = 365.2422

# Consider an area which has tides of the three periods,
# such that the absolute maximum tidal range is 2.00m
# (i.e. Amplitude of 1m, because the range is double the amplitude for a wave)
MaxTrange = 2.00
Amax = 0.5 * MaxTrange
# assuming half-daily, monthly and annual tides account for 70%, 20% and 10 %
# respectively of the tidal signal, assign the amplitude for these three periods
A_1 = 0.7*Amax
A_2 = 0.2*Amax
A_3 = 0.1*Amax

# rather than assume that each cycle starts on 1 Jan, create an offset
# for each of the tide component
# (units: radians, therefore choose offsets between 0 and 2pi, i.e. fraction of each cycle)
phi_1 = 0
phi_2 = 0.3*np.pi
phi_3 = 0.7*np.pi

# plot the tide height over 3 years
# first, define a timevector, in days. time step is 0.2 x shortest period
dt = 0.2*min(Tp_1,Tp_2,Tp_3)
T_vec = np.arange(0,3*365,dt)
# calculate the tide height, assuming sine curve.
# Note that period Tp in days must be shifted into angular frequency, so that
# omega * t is in units of radians (i.e. 2 pi = 1 cycle)
#  if Tp is in units seconds, f=1/Tp gives the frequency in Hz (cycle per second),
# then multipy by 2 pi to convert cycles per second to radians/second.

# In this case, it makes sense to keep both Tp (period) and time (T_vec)
# in units of days (rather than convert to seconds and back again):
# and 2*pi/Tp is radians per day, so (2*pi/Tp)*Tvec is in radians, as required

# tide height at time t is therefore calcuated from
# amplitude, period/frequency and phase shift as follows
# for the three components of the signal, using y=(omega * t+ phi)
Y_1 = A_1*np.sin((2*np.pi/Tp_1)*T_vec+phi_1)
Y_2 = A_2*np.sin((2*np.pi/Tp_2)*T_vec+phi_2)
Y_3 = A_3*np.sin((2*np.pi/Tp_3)*T_vec+phi_3)
# now add sealevel rise of 1.5 mm/year, converted to m/day and multiplied by the time vector
SLR = 1.5e-3/365.25
Y_4 = T_vec*SLR
# actual tide height is the sum of the contributions from daily, monthly and annual cycles
Y = Y_1+Y_2+Y_3+Y_4

# put the results together in a dataframe for plotting
tide_data = pd.DataFrame(data={'y1': Y_1, 'y2': Y_2,'y3': Y_3,'y': Y}, index = pd.to_datetime(T_vec,unit='D'))
# plot the results, using different axis scales to see on different timescales

fig,ax = plt.subplots()
ax.plot(tide_data.index,tide_data['y'])
ax.set_xlabel('Days')
ax.set_ylabel('Tide height (m)')
ax.set_title('One week')

fig,ax = plt.subplots()
ax.plot(tide_data.index,tide_data['y1'],'r')
ax.plot(tide_data.index,tide_data['y2'],'b')
ax.plot(tide_data.index,tide_data['y3'],'k')
ax.set_xlabel('Days')
ax.set_ylabel('Tide height (m)')
ax.set_title('One week')

# fig,ax = plt.subplots()
# ax.plot(tide_data.index,tide_data['y'])
# ax.set_xlabel('Days')
# ax.set_ylabel('Tide height (m)')
# ax.set_title('One month')

# fig,ax = plt.subplots()
# ax.plot(tide_data.index,tide_data['y1'],'r')
# ax.plot(tide_data.index,tide_data['y2'],'b')
# ax.plot(tide_data.index,tide_data['y3'],'k')
# ax.set_xlabel('Days')
# ax.set_ylabel('Tide height (m)')
# ax.set_title('One year')

# fft of the tidal signal
fA = np.fft.fft(Y)
# determine the corresponding frequency

plot_acf(Y)

plot_acf(np.diff(Y))
seasonal_decompose(tide_data['y'].resample('D').mean().ffill()).plot()
