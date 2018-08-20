#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:53:33 2018

@author: vedanshu
"""

from matplotlib import rc
import matplotlib.pyplot as plt
import json
import numpy as np

def plot():
    state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
    for st in state.values():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rc('font',**{'family':'serif','serif':['Times New Roman']})
        ax.set_title("MAPE for state "+ st)
        
        for fold in np.arange(1,6):
                fname1 = "results_" + st + "_TS_" + str(fold) + "CSO"
                fname2 = "results_" + st + "_TS_" + str(fold) + "GD"
                _, e_mape1, e_rmse1, e_mae1, _, _, _, _ = json.load(open(fname1))
                _, e_mape2, e_rmse2, e_mae2, _, _, _, _ = json.load(open(fname2))
                label = "CSO Split "+str(fold)
                ax.plot(np.arange(0, 1500), [e for e in e_mape1], label=label )
                ax.plot(np.arange(0, 1500), [e for e in e_mape2],':', label = "GD Split "+str(fold))
                ax.set_ylim([10,20])
                ax.set_xlim([200,1500])
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MAPE')
                ax.grid(True)
        plt.legend(loc="center", bbox_to_anchor=(0.5, -0.4), ncol=3)
        plt.show()
