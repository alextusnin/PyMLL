#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:34:10 2022

@author: alextusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys,os
curr_dir = os.getcwd()
PyMLL_dir = os.path.dirname(curr_dir)
sys.path.append(PyMLL_dir)
sys.path.append('/home/tusnin/Documents/Projects/PyCORe')
import PyCORe_main as pcm

import PyMLL_main as PyMLL
import time
from scipy.constants import c,hbar


FSR = 50e9
D1 = 2*np.pi*FSR
Num_of_modes = 2**12
D2 = 2*np.pi*0.5*1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = -2*np.pi*25e3*0
D4 = -2*np.pi*1e3*0
D5 = 2*np.pi*54*0
D6 = -2*np.pi*1*0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)

Dint = (mu**2*D2/2 + mu**3*D3/6+mu**4*D4/24+mu**5*D5/120+mu**6*D6/720)
kappa_ex_ampl = 50e6*2*np.pi
kappa_ex = kappa_ex_ampl*np.ones([Num_of_modes])
nn=10000
# gain = PyMLL.Gain()
# gain.Transform_to_rad_Hz(1.9)
# a1,a2,w1,w2,sigma1,sigma2=gain.FitGaussian()


# nu0 = gain.frequency[np.argmax(gain.gain_data)]

# frequency_grid = nu0+ FSR*mu

# plt.plot(gain.frequency,gain.gain_rad_Hz/2/np.pi/1e6)
# plt.plot(frequency_grid,(a1*np.exp(-(frequency_grid-w1)**2/2/sigma1**2)+a2*np.exp(-(frequency_grid-w2)**2/2/sigma2**2))/2/np.pi/1e6,'.')

PhysicalParameters = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : FSR ,
                      'width' : 2.2*1e-6,
                      'height' : 0.81*1e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint}
GLE = PyMLL.GLE()
GLE.Init_From_Dict(PhysicalParameters)

simulation_parameters = {'slow_time' : 1e-8,
                         'Number of points' : nn,
                         'noise_level' : 1e-7,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-12,
                         'relative_tolerance' : 1e-12,
                         'max_internal_steps' : 2000}
#%%
G0 = GLE.gain.FitGaussian()[0]/2
G2 = (1/GLE.gain.FitGaussian()[2]**2)*D1**2*G0
C=0; Phi = 0;
A0 = np.sqrt(3/4*(G0-GLE.kappa.max())/(GLE.g0)*abs(D2)/G2)*np.sqrt(hbar*GLE.w0)
B = np.sqrt(3/G2/(1+C**2)*(G0-GLE.kappa.max()))
Soliton = np.complex_(A0*1/(np.cosh(B*(GLE.phi-np.pi)))**(1+1j*C))
SolSpectrum = np.zeros(Num_of_modes,dtype=complex)
#SolSpectrum[0] = np.sqrt(0.0008)
SolSpectrum = np.fft.fft(Soliton)*Num_of_modes#/1e10#/Num_of_modes**2/
#SolSpectrum = np.zeros(Num_of_modes,dtype=complex)
#SolSpectrum=np.load('./map2d.npy')[0,:]
# A0 = 0.00001
# sigma = 0.1
# Soliton = A0*1/(np.cosh((GLE.phi-np.pi)/sigma)) #A0*np.exp(-(GLE.phi-np.pi)**2/2/sigma)
# SolSpectrum = np.fft.fft(Soliton)#/1e10#/1e10#/Num_of_modes**2/
#%%
start_time = time.time()
# map2d=GLE.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,Seed=SolSpectrum,dt=1e-5)
map2d=GLE.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,Seed=SolSpectrum,dt=1e-14)
# map2d=GLE.Propagate_PseudoSpectralStiffCLIB(simulation_parameters,Seed=SolSpectrum,dt=1e-2)
#map2d=GLE.Propagate_PseudoSpectralStiffCLIB(simulation_parameters,Seed=SolSpectrum,Solver='Sie',dt=1e-2)#*np.sqrt((hbar*GLE.w0))
print("--- %s seconds ---" % (time.time() - start_time))
#%%
start_time = time.time()
for jj in range(0):
    Seed=map2d[-1,:]
    simulation_parameters['noise_level']=0
    #map2d=GLE.Propagate_PseudoSpectralStiffCLIB(simulation_parameters,Seed=Seed,dt=1e-3)
    map2d=GLE.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,Seed=Seed,dt=1e-14)
    print("--- %s seconds ---" % (time.time() - start_time))
    
#print("--- %s seconds ---" % (time.time() - start_time))
#%%
#pcm.Plot_Map(np.fft.ifft(map2d[:,:],axis=1),np.arange(nn))
#%%
#plt.plot(np.arange(nn)*simulation_parameters['slow_time']/nn,np.sum(np.abs(map2d[:,:])**2,axis=1)*((hbar*GLE.w0))/GLE.gain.P_th)
plt.plot(np.arange(nn)*simulation_parameters['slow_time']/nn,np.sum(np.abs(map2d[:,:])**2,axis=1)/GLE.gain.P_th)
#%%
plt.figure()
plt.pcolormesh(GLE.frequency_grid,np.arange(nn),np.log(abs(np.fft.fftshift(np.fft.ifft(map2d[:,:],axis=0)))**2/1e-3),)
#%%
# plt.figure()
# plt.plot(GLE.frequency_grid,np.fft.fftshift(GLE.gain_grid-GLE.kappa))
# plt.plot(GLE.gain.frequency,GLE.gain.gain_rad_Hz)
#%%
#%%

# data_dir = '/home/alextusnin/Documents/MLL/simulations'
# try:
#     dir_2_save = data_dir+'/data/'+str(np.round(FSR/1e9,1))+'_'+str(np.round(D2/2/np.pi/1e6,1))+ '_fancy_collision/'
#     os.mkdir(dir_2_save)
# except:
#     pass
# GLE.Save_Data(map2d,simulation_parameters,dir_2_save)    