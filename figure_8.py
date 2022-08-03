import os
import argparse
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle

TMs = 50

if __name__ == "__main__":
    # This script is to obtain the figure 8 from COMNET 2022 paper. We plot the boxplots
    # of SAP, Enero and DEFO for each number of link failures.

    # Before executing this file we must execute the eval_on_zoo_topologies.py script
    # to evaluate the DRL agent over the link failure topologies. Also before that, we must create
    # the link failure dataset (execute generate_link_failure_topologies.py). We must create a dataset
    # for each topology where we want to try the link failure scenario and also execute the eval_on_zoo_topologies.py script
    # on each of these topologies before I execute this script.
    # python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_HurricaneElectric
    # python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_Janetbackbone
    # python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_EliBackbone

    parser = argparse.ArgumentParser(description='Parse files and create plots')

    # The flag 'p' points to the folder where the .pckl files are found
    parser.add_argument('-d', help='differentiation string for the model', type=str, required=True, nargs='+')
    parser.add_argument('-f', help='folder directory', type=str, required=True, nargs='+')
    parser.add_argument('-num_topologies', help='How many NEW topologies we want', type=int, required=True, nargs='+')

    args = parser.parse_args()

    differentiation_str = args.d[0]
    folder_dir = args.f[0]
    num_topologies_x_link_failure = args.num_topologies[0]

    fail_2 = np.zeros(num_topologies_x_link_failure*TMs)
    fail_4 = np.zeros(num_topologies_x_link_failure*TMs)
    fail_6 = np.zeros(num_topologies_x_link_failure*TMs)
    fail_8 = np.zeros(num_topologies_x_link_failure*TMs)

    sap_fail_2 = np.zeros(num_topologies_x_link_failure*TMs)
    sap_fail_4 = np.zeros(num_topologies_x_link_failure*TMs)
    sap_fail_6 = np.zeros(num_topologies_x_link_failure*TMs)
    sap_fail_8 = np.zeros(num_topologies_x_link_failure*TMs)

    defo_fail_2 = np.zeros(num_topologies_x_link_failure*TMs)
    defo_fail_4 = np.zeros(num_topologies_x_link_failure*TMs)
    defo_fail_6 = np.zeros(num_topologies_x_link_failure*TMs)
    defo_fail_8 = np.zeros(num_topologies_x_link_failure*TMs)

    filename_list = []
    X_axis = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    path_to_dir = "./Images/EVALUATION/"+differentiation_str+'/'

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    axis_counter = 0
    topology_Name = ""
    directory = folder_dir+'/'+differentiation_str
    # Iterate over all topologies and evaluate our DRL agent on all TMs
    for filename in os.listdir(directory):
        dir_to_topology_rewards = directory+"/"+filename
    
        for subdir, dirs, files in os.walk(dir_to_topology_rewards):
            for file in files:
                if file.endswith((".pckl")):
                    results = []
                    path_to_pckl_rewards = dir_to_topology_rewards + '/'

                    with open(path_to_pckl_rewards+file, 'rb') as f:
                        results = pickle.load(f)
                    if topology_Name=="":
                        topology_Name = file.split('_')[0]
                    topology_id = int(file.split('_')[2].split('.')[0])-1
                    num_links_failure = int(file.split('_')[1])
                    tm_id = int(file.split('.')[1])

                    if num_links_failure==1:
                        fail_2[topology_id*TMs+tm_id] = results[3]
                        sap_fail_2[topology_id*TMs+tm_id] = results[8]
                        defo_fail_2[topology_id*TMs+tm_id] = results[1]
                    elif num_links_failure==2:
                        fail_4[topology_id*TMs+tm_id] = results[3]
                        sap_fail_4[topology_id*TMs+tm_id] = results[8]
                        defo_fail_4[topology_id*TMs+tm_id] = results[1]
                    elif num_links_failure==3:
                        fail_6[topology_id*TMs+tm_id] = results[3]
                        sap_fail_6[topology_id*TMs+tm_id] = results[8]
                        defo_fail_6[topology_id*TMs+tm_id] = results[1]
                    elif num_links_failure==4:
                        fail_8[topology_id*TMs+tm_id] = results[3]
                        sap_fail_8[topology_id*TMs+tm_id] = results[8]
                        defo_fail_8[topology_id*TMs+tm_id] = results[1]

    # Make all boxplots in one
    lk_fail = [2, 4, 6, 8]

    # This is to compute the average of 50 TMs and for Enero and for DEFO
    dd2_aux = np.zeros((TMs,3))
    dd4_aux = np.zeros((TMs,3))
    dd6_aux = np.zeros((TMs,3))
    dd8_aux = np.zeros((TMs,3))

    dd2 = pd.DataFrame(columns=['Enero','DEFO', 'SAP', 'Number Link Failures'])
    dd4 = pd.DataFrame(columns=['Enero','DEFO', 'SAP', 'Number Link Failures'])
    dd6 = pd.DataFrame(columns=['Enero','DEFO', 'SAP', 'Number Link Failures'])
    dd8 = pd.DataFrame(columns=['Enero','DEFO', 'SAP', 'Number Link Failures'])

    for num_fail in lk_fail:
        it = 0
        for topology_id in range(num_topologies_x_link_failure):
            dd2_aux.fill(0.0)
            dd4_aux.fill(0.0)
            dd6_aux.fill(0.0)
            dd8_aux.fill(0.0)
            for tm_id in range(TMs):
                if num_fail==2:
                    dd2_aux[tm_id,0]=fail_2[topology_id*TMs+tm_id]
                    dd2_aux[tm_id,1]=defo_fail_2[topology_id*TMs+tm_id]
                    dd2_aux[tm_id,2]=sap_fail_2[topology_id*TMs+tm_id]
                if num_fail == 4:
                    dd4_aux[tm_id,0]=fail_4[topology_id*TMs+tm_id]
                    dd4_aux[tm_id,1]=defo_fail_4[topology_id*TMs+tm_id]
                    dd4_aux[tm_id,2]=sap_fail_4[topology_id*TMs+tm_id]
                if num_fail == 6:
                    dd6_aux[tm_id,0]=fail_6[topology_id*TMs+tm_id]
                    dd6_aux[tm_id,1]=defo_fail_6[topology_id*TMs+tm_id]
                    dd6_aux[tm_id,2]=sap_fail_6[topology_id*TMs+tm_id]
                if num_fail == 8:
                    dd8_aux[tm_id,0]=fail_8[topology_id*TMs+tm_id]
                    dd8_aux[tm_id,1]=defo_fail_8[topology_id*TMs+tm_id]
                    dd8_aux[tm_id,2]=sap_fail_8[topology_id*TMs+tm_id]
            
            if num_fail==2:
                dd2.loc[it] = [np.mean(dd2_aux[:,0]), np.mean(dd2_aux[:,1]), np.mean(dd2_aux[:,2]), num_fail]
            if num_fail == 4:
                dd4.loc[it] = [np.mean(dd4_aux[:,0]), np.mean(dd4_aux[:,1]), np.mean(dd4_aux[:,2]), num_fail]
            if num_fail == 6:
                dd6.loc[it] = [np.mean(dd6_aux[:,0]), np.mean(dd6_aux[:,1]), np.mean(dd6_aux[:,2]), num_fail]
            if num_fail == 8:
                dd8.loc[it] = [np.mean(dd8_aux[:,0]), np.mean(dd8_aux[:,1]), np.mean(dd8_aux[:,2]), num_fail]
            it += 1
    
    # Define some hatches
    hatches = cycle(['\\', 'O', '/'])
    cdf = pd.concat([dd2,dd4,dd6,dd8])
    mdf = pd.melt(cdf, id_vars=['Number Link Failures'], var_name=['Topology'])      # MELT
    ax = sns.boxplot(x="Number Link Failures", y="value", hue="Topology", data=mdf, palette="mako")  # RUN PLOT
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = (3.47, 2.0)
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    ax.set_xlabel("Number of Link Failures",fontsize=16)
    ax.set_ylabel("Maximum Link Utilization",fontsize=16)
    plt.rcParams["axes.labelweight"] = "bold"
    ax.grid(which='major', axis='y', linestyle='-')
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['pdf.fonttype'] = 42
    # Loop over the bars
    for i, patch in enumerate(ax.artists):
        # Boxes from left to right
        hatch = next(hatches)
        patch.set_hatch(hatch*2)
        col = patch.get_facecolor()
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
        patch.set_edgecolor("black")
        patch.set_facecolor('None')
    
    plt.legend(loc='upper left')
    plt.ylim((0.4, 1.8))
    plt.tight_layout()
    plt.savefig(path_to_dir+'/Figure_8_'+topology_Name+'.pdf', bbox_inches='tight',pad_inches = 0)