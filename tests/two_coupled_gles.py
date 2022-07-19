#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:32:15 2022

@author: alextusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys,os
curr_dir = os.getcwd()
PyMLL_dir = os.path.dirname(curr_dir)
sys.path.append(PyMLL_dir)
sys.path.append('/home/alextusnin/Documents/Projects/PyCORe')
import PyCORe_main as pcm

import PyMLL_main as PyMLL
import time
from scipy.constants import c,hbar



FSR = 50e9


Num_of_modes = 2**10
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
D1_1 = 2*np.pi*FSR
D2_1 = 2*np.pi*4*1e5#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
##D1_2 = 2*np.pi*FSR
#D2_2 = -2*np.pi*0.5*1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D1_2 = D1_1
D2_2 = D2_1

Dint_1 = (mu**2*D2_1/2 )
Dint_2 = (mu**2*D2_2/2 )
kappa_ex_1_ampl = 20e6*2*np.pi
kappa_ex_1 = kappa_ex_1_ampl*np.ones([Num_of_modes])
kappa_ex_2_ampl = 50e6*2*np.pi
kappa_ex_2 = kappa_ex_2_ampl*np.ones([Num_of_modes])
J = 1e7*2*np.pi
nn=10000

PhysicalParameters_1 = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : FSR ,
                      'width' : 2.2*1e-6,
                      'height' : 0.81*1e-6,
                      'kappa_0' : 20e6*2*np.pi,
                      'kappa_ex' : kappa_ex_1,
                      'Dint' : Dint_1}
PhysicalParameters_2 = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : FSR ,
                      'width' : 2.2*1e-6,
                      'height' : 0.81*1e-6,
                      'kappa_0' : 20e6*2*np.pi,
                      'kappa_ex' : kappa_ex_2,
                      'Dint' : Dint_2}
GLE_1 = PyMLL.GLE()
GLE_1.Init_From_Dict(PhysicalParameters_1)

GLE_2 = PyMLL.GLE()
GLE_2.InitZeroGain()
GLE_2.Init_From_Dict(PhysicalParameters_2)


TwoGLEs = PyMLL.TwoCoupledGLE(GLE_1,GLE_2,J)


#%%

simulation_parameters = {'slow_time' : 1e-8,
                         'Number of points' : nn,
                         'noise_level' : 1e-7,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-14,
                         'relative_tolerance' : 1e-14,
                         'max_internal_steps' : 2000}

#%%

map2d=TwoGLEs.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,dt=1e-14)

#%%
start_time = time.time()
for jj in range(500):
    Seed=map2d[-1,:,:]
    simulation_parameters['noise_level']=0
    map2d=TwoGLEs.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,Seed=Seed,dt=1e-14,HardSeed=True)
    print("--- %s seconds ---" % (time.time() - start_time))
#%%
plt.plot(np.arange(nn)*simulation_parameters['slow_time']/nn,np.sum(np.abs(map2d[:,:,0])**2,axis=1)/GLE_1.P_th)
plt.plot(np.arange(nn)*simulation_parameters['slow_time']/nn,np.sum(np.abs(map2d[:,:,1])**2,axis=1)/GLE_1.P_th)

#%%
#pcm.Plot_Map(np.fft.ifft(map2d[:,:,0],axis=1),np.arange(nn))
#%%
slow_freq = (np.arange(0,nn) - nn/2)/simulation_parameters['slow_time']
plt.figure()
plt.pcolormesh(GLE_1.frequency_grid,slow_freq/1e9,10*np.log10(abs(np.fft.fftshift(np.fft.ifft(map2d[:,:,1],axis=0)))**2/1e-3),cmap='afmhot',vmin=-150)

#%%
plt.plot(GLE_1.frequency_grid,np.fft.fftshift(GLE_1.gain_grid-GLE_1.kappa))

#%%

data_dir = '/home/alextusnin/Documents/MLL/simulations'
dir_2_save = data_dir+'/data_coupled/'+str(np.round(FSR/1e9,1))+'_'+str(np.round(D2_1/2/np.pi/1e6,1))+'_' +str(np.round(J/2/np.pi/1e9,2)) + '/'
try:
    
    os.mkdir(dir_2_save)
except:
    pass
#%%%
TwoGLEs.Save_Data(map2d,simulation_parameters,dir_2_save)    