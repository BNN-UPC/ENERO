import argparse
import logging
import os
import sys
import json
import pandas as pd
import seaborn as sns
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from operator import add, sub
import operator
import pickle
from scipy.signal import savgol_filter

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

#folders = ["../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_NEW_Garr199905/EVALUATE/"]
folders = ["../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/evalRes_NEW_EliBackbone/EVALUATE/","../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/evalRes_NEW_Janetbackbone/EVALUATE/","../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/evalRes_NEW_HurricaneElectric/EVALUATE/"]

if __name__ == "__main__":
    # This script is to plot the Figures 5 and 6 from COMNET 2022 paper.

    # Before executing this file we must execute the eval_on_single_topology.py file to evaluate the DRL model and store the results
    # We also need to evaluate DEFO for these new topologies. To do this, I copy the corresponding 
    # folder where it needs to be and I execute the script run_Defo_single_top.py for each topology.
    # python figures_5_and_6.py -d SP_3top_15_B_NEW 
    parser = argparse.ArgumentParser(description='Parse files and create plots')

    # The flag 'd' indicates the directory where to store the figures
    parser.add_argument('-d', help='differentiation string for the model', type=str, required=True, nargs='+')

    args = parser.parse_args()

    differentiation_str = args.d[0]

    drl_top1_uti = []
    ls_top1_uti = []
    enero_top1_uti = []
    cost_drl_top1 = []
    cost_ls_top1 = []
    cost_enero_top1 = []

    drl_top2_uti = []
    ls_top2_uti = []
    enero_top2_uti = []
    cost_drl_top2 = []
    cost_ls_top2 = []
    cost_enero_top2 = []

    drl_top3_uti = []
    ls_top3_uti = []
    enero_top3_uti = []
    cost_drl_top3 = []
    cost_ls_top3 = []
    cost_enero_top3 = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    path_to_dir = "./Images/EVALUATION/"+differentiation_str+'/'

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    dd_Eli = pd.DataFrame(columns=['DRL','LS','Enero','Topologies'])
    dd_Janet = pd.DataFrame(columns=['DRL','LS','Enero','Topologies'])
    dd_Hurricane = pd.DataFrame(columns=['DRL','LS','Enero','Topologies'])

    # Iterate over all topologies and evaluate our DRL agent on all TMs
    for folder in folders:
        drl_eval_res_folder = folder+differentiation_str+'/'
        topology_eval_name = folder.split('NEW_')[1].split('/')[0]
        for subdir, dirs, files in os.walk(drl_eval_res_folder):
            it = 0
            for file in files:
                if file.endswith((".pckl")):
                    results = []
                    path_to_pckl_rewards = drl_eval_res_folder + topology_eval_name + '/'
                    with open(path_to_pckl_rewards+file, 'rb') as f:
                        results = pickle.load(f)
                    if folder==folders[0]:
                        dd_Eli.loc[it] = [results[9],results[7],results[3],topology_eval_name]
                        cost_ls_top1.append(results[15])
                        cost_drl_top1.append(results[14])
                        cost_enero_top1.append(results[16])
                    elif folder==folders[1]:
                        dd_Janet.loc[it] = [results[9],results[7],results[3],topology_eval_name]
                        cost_ls_top2.append(results[15])
                        cost_drl_top2.append(results[14])
                        cost_enero_top2.append(results[16])
                    else:
                        dd_Hurricane.loc[it] = [results[9],results[7],results[3],topology_eval_name]
                        cost_ls_top3.append(results[15])
                        cost_drl_top3.append(results[14])
                        cost_enero_top3.append(results[16])
                    it += 1
    
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['figure.figsize'] = (11.5, 9)
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 17
    fig, ax = plt.subplots()
    
    n = np.arange(1,len(cost_ls_top1)+1) / np.float(len(cost_ls_top1))
    Xs = np.sort(cost_ls_top1)
    ax.step(Xs,n, c='cyan', linestyle=(0, (1,1)), label="LS EliBackbone", linewidth=4) 
    Xs = np.sort(cost_drl_top1)
    ax.step(Xs,n,c='darkgreen', linestyle='-', label="DRL EliBackbone", linewidth=4) 
    Xs = np.sort(cost_enero_top1)
    ax.step(Xs,n,c='maroon', linestyle=(0, (2.5, 1)),label="Enero EliBackbone", linewidth=4) 
    Xs = np.sort(cost_ls_top2)
    ax.step(Xs,n, c='dodgerblue', linestyle=(0, (1, 2.5)),label="LS Janetbackbone", linewidth=4) 
    Xs = np.sort(cost_drl_top2)
    ax.step(Xs,n,c='lime', linestyle='-',label="DRL Janetbackbone", linewidth=4) 
    Xs = np.sort(cost_enero_top2)
    ax.step(Xs,n,c='red', linestyle=(0, (2.5, 3)),label="Enero Janetbackbone", linewidth=4) 
    Xs = np.sort(cost_ls_top3)
    ax.step(Xs,n, c='navy', linestyle=(0, (1,6)),label="LS HurricaneElectric", linewidth=4) 
    Xs = np.sort(cost_drl_top3)
    ax.step(Xs,n,c='palegreen', linestyle='-',label="DRL HurricaneElectric", linewidth=4) 
    Xs = np.sort(cost_enero_top3)
    ax.step(Xs,n,c='orange', linestyle=(0, (2.5, 6)),label="Enero HurricaneElectric", linewidth=4) 

    plt.ylim((0, 1.005))
    plt.xlim((0, 50.0))
    plt.xticks(np.arange(0, 50, 8))
    plt.ylabel('CDF', fontsize=22)
    plt.xlabel("Execution Cost (s)", fontsize=20)
    plt.grid(color='gray')
    plt.legend(loc='lower right', ncol=3, bbox_to_anchor=(1.03, -0.3))
    plt.tight_layout()
    plt.savefig(path_to_dir+'Figure_6.pdf', bbox_inches='tight',pad_inches = 0)
    plt.close()

 
    # Define some hatches
    hatches = cycle(['-', '|', ''])
    cdf = pd.concat([dd_Eli,dd_Janet,dd_Hurricane])
    mdf = pd.melt(cdf, id_vars=['Topologies'], var_name=['Topology'])      # MELT
    ax = sns.boxplot(x="Topologies", y="value", hue="Topology", data=mdf, palette="mako")  # RUN PLOT
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = (3.47, 2.0)
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 24
    ax.set_xlabel("",fontsize=0)
    ax.set_ylabel("Maximum Link Utilization",fontsize=24)
    plt.rcParams["axes.labelweight"] = "bold"
    ax.grid(which='major', axis='y', linestyle='-')
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['pdf.fonttype'] = 42
    # Loop over the bars
    for i, patch in enumerate(ax.artists):
        # Boxes from left to right
        hatch = next(hatches)
        patch.set_hatch(hatch*2)
        col = patch.get_facecolor()
        #patch.set_edgecolor(col)
        patch.set_edgecolor("black")
        patch.set_facecolor('None')

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color("black")
            line.set_mfc("black")
            line.set_mec("black")
            # Change color of the median
            if j == i*6+4:
                line.set_color("orange")
                line.set_mfc("orange")
                line.set_mec("orange")

    for i, patch in enumerate(ax.patches):
        hatch = next(hatches)
        patch.set_hatch(hatch*2)
        col = patch.get_facecolor()
        #patch.set_edgecolor(col)
        patch.set_edgecolor("black")
        patch.set_facecolor('None')
    
    plt.legend(loc='upper left', ncol=3)
    plt.ylim((0.5, 1.2))
    plt.tight_layout()
    plt.savefig(path_to_dir+'Figure_5.pdf', bbox_inches='tight',pad_inches = 0)
    plt.clf()
    plt.close()
