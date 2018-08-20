#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 11:04:09 2018

@author: vedanshu
"""

import numpy as np
import json

def export(cso = False):
    state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
    error_mape, error_rmse, error_mae = [], [], []
    
    for st in state.values():
        error_state_mape, error_state_rmse, error_state_mae = [], [], []
        for fold in np.arange(1,6):
            if cso:
                fname = "results_" + st + "_TS_" + str(fold) + "CSO"
            else:
                fname = "results_" + st + "_TS_" + str(fold) + "GD"
            _, e_mape, e_rmse, e_mae, _, _, _, _ = json.load(open(fname))
            error_state_mape.append(e_mape[-1])
            error_state_rmse.append(e_rmse[-1])
            error_state_mae.append(e_mae[-1])
        error_mape.append(error_state_mape)
        error_rmse.append(error_state_rmse)
        error_mae.append(error_state_mae)
       
    if cso:
        np.savetxt('results_TS_MAPE_CSO.csv', np.vstack(error_mape), delimiter = ',')
        np.savetxt('results_TS_RMSE_CSO.csv', np.vstack(error_rmse), delimiter = ',')
        np.savetxt('results_TS_MAE_CSO.csv', np.vstack(error_mae), delimiter = ',')
    else:
        np.savetxt('results_TS_MAPE_GD.csv', np.vstack(error_mape), delimiter = ',')
        np.savetxt('results_TS_RMSE_GD.csv', np.vstack(error_rmse), delimiter = ',')
        np.savetxt('results_TS_MAE_GD.csv', np.vstack(error_mae), delimiter = ',')
    
    error_mape, error_rmse, error_mae = [], [], []
    for st in state.values():
        error_state_mape, error_state_rmse, error_state_mae = [], [], []
        for fold in np.arange(1,6):
            if cso:
                fname = "results_" + st + "_5Fold_" + str(fold) + "CSO"
            else:
                fname = "results_" + st + "_5Fold_" + str(fold) + "GD"
            _, e_mape, e_rmse, e_mae, _, _, _, _ = json.load(open(fname))
            error_state_mape.append(e_mape[-1])
            error_state_rmse.append(e_rmse[-1])
            error_state_mae.append(e_mae[-1])
        error_mape.append(error_state_mape)
        error_rmse.append(error_state_rmse)
        error_mae.append(error_state_mae)
       
    if cso:
        np.savetxt('results_fold_MAPE_CSO.csv', np.vstack(error_mape), delimiter = ',')
        np.savetxt('results_fold_RMSE_CSO.csv', np.vstack(error_rmse), delimiter = ',')
        np.savetxt('results_fold_MAE_CSO.csv', np.vstack(error_mae), delimiter = ',')
    else:
        np.savetxt('results_fold_MAPE_GD.csv', np.vstack(error_mape), delimiter = ',')
        np.savetxt('results_fold_RMSE_GD.csv', np.vstack(error_rmse), delimiter = ',')
        np.savetxt('results_fold_MAE_GD.csv', np.vstack(error_mae), delimiter = ',')
