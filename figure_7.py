import argparse
import logging
import os
import sys
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from operator import add, sub
import operator
import pickle
from scipy.signal import savgol_filter

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump


if __name__ == "__main__":
    # This script is to obtain the figure 7 where for each TM we see it's progress in Enero's optimization. 
    # Figure 7 from COMNET 2022 paper.
    
    # Before executing this file we must execute the eval_on_single_topology.py file to evaluate the DRL model and store the results
    # python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_EliBackbone/EVALUATE/ -t EliBackbone
    # python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_Janetbackbone/EVALUATE/ -t Janetbackbone
    # python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_HurricaneElectric/EVALUATE/ -t HurricaneElectric
    # Then, we execute the script like:
    # python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_EliBackbone/EVALUATE/ -t EliBackbone
    parser = argparse.ArgumentParser(description='Parse files and create plots')

    # The flag 'p' points to the folder where the .pckl files are found
    parser.add_argument('-p', help='data folder', type=str, required=True, nargs='+')
    parser.add_argument('-d', help='differentiation string for the model', type=str, required=True, nargs='+')
    parser.add_argument('-t', help='topology name', type=str, required=True, nargs='+')

    args = parser.parse_args()

    differentiation_str = args.d[0]
    drl_eval_res_folder = args.p[0]+differentiation_str+'/'
    topology_eval_name = args.t[0]

    list_utis = []
    list_timers = []
    avg_utis_drl = []
    avg_timers_drl = []
    avg_utis_init = []
    avg_timers_init = []
    avg_utis_end = []
    avg_timers_end = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    path_to_dir = "./Images/EVALUATION/"+differentiation_str+'/'

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    plt.rcParams.update({'font.size': 12})
    # Iterate over all topologies and evaluate our DRL agent on all TMs
    for subdir, dirs, files in os.walk(drl_eval_res_folder):
        for file in files:
            if file.endswith((".timesteps")):
                results = []
                list_utis = []
                list_timers = []

                path_to_timesteps = drl_eval_res_folder + topology_eval_name + '/'+file
                time_samples = json.load(open(path_to_timesteps))
                #time_samples = sorted(time_samples, key=lambda tup: tup[0], reverse=False)
                for elem in time_samples:
                    list_timers.append(float(elem[0])) # time
                    list_utis.append(float(elem[1])) # uti
                plt.plot(list_timers, list_utis, color="grey", alpha=0.3)  

                tm_id = file.split('.')[1] 
                results_file = drl_eval_res_folder + topology_eval_name + '/'+topology_eval_name+'.'+tm_id+'.pckl'
                results = []
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)

                avg_utis_drl.append(results[9])
                avg_timers_drl.append(results[14])
                avg_timers_init.append(time_samples[0][0])
                avg_utis_init.append(time_samples[0][1])
                avg_timers_end.append(time_samples[len(time_samples)-1][0])
                avg_utis_end.append(time_samples[len(time_samples)-1][1])
    # Plot again the last TM to show the label
    plt.plot(list_timers, list_utis, color="grey", label="TMs", alpha=0.3)   
    init = plt.scatter(np.mean(avg_timers_init), np.mean(avg_utis_init), label="Mean Maximum Link Utilization OSPF", color="navy", marker='X', s=150, zorder=2.5)     
    plt.scatter(np.mean(avg_timers_drl), np.mean(avg_utis_drl), label="Mean Maximum Link Utilization DRL", color="navy", marker='o', s=150, zorder=2.5)     
    end = plt.scatter(np.mean(avg_timers_end), np.mean(avg_utis_end), label="Mean Maximum Link Utilization Enero", color="navy", marker='^', s=150, zorder=2.5) 
    plt.ylabel('Maximum Link Utilization', fontsize=15)
    plt.legend()
    plt.xlabel("Time (s)", fontsize=15)
    plt.tight_layout()
    plt.savefig(path_to_dir+'Figure_7_'+topology_eval_name+'.pdf', bbox_inches = 'tight',pad_inches = 0)