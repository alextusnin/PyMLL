#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:49:18 2022

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

data_dir = '/home/alextusnin/Documents/MLL/simulations/data/10.0_1.0_single_soliton/'

map2d_scan = np.zeros([],dtype=complex)#np.load('map2d_scan.npy')
simulation_parameters={}

GLE=PyMLL.GLE()
simulation_parameters,map2d_scan=GLE.Init_From_File(data_dir)
Seed=map2d_scan[-1,:]

#%%
# for jj in range(50):
#     map2d=GLE.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,Seed=Seed,dt=1e-5)
#     Seed=map2d[-1,:]

# #%%
# np.save(data_dir +'map2d_50us.npy',map2d)