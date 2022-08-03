import os
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def read_defo_optim_time(standard_out_file):
    optim_time = 0
    with open(standard_out_file) as fd:
        while (True):
            line = fd.readline()
            if line.startswith("optimization time"):
                camps = line.split(":")
                optim_time = float(camps[-1].split('\n')[0])
                break
        return float(optim_time)/1000

def read_max_load_link(standard_out_file):
    pre_optim_max_load_link, post_optim_max_load_link = 0, 0
    with open(standard_out_file) as fd:
        while (True):
            line = fd.readline()
            if line.startswith("pre-optimization"):
                camps = line.split(" ")
                pre_optim_max_load_link = float(camps[-1].split('\n')[0])
            elif line.startswith("post-optimization"):
                camps = line.split(" ")
                post_optim_max_load_link = float(camps[-1].split('\n')[0])
                break
        return (pre_optim_max_load_link, post_optim_max_load_link)

if __name__ == "__main__":
    # This script is to obtain the figure 10 from Conext 2021 paper.

    # Before executing this file we must execute the eval_on_zoo_topologies.py file to evaluate the DRL model and store the results
    # python figure_9.py -d SP_3top_15_B_NEW -p ../rwds-results-1-link_capacity-unif-05-1-zoo

    parser = argparse.ArgumentParser(description='Parse files and create plots')

    # The flag 'p' points to the folder where the .pckl files are found
    parser.add_argument('-p', help='data folder', type=str, required=True, nargs='+')
    parser.add_argument('-d', help='differentiation string for the model', type=str, required=True, nargs='+')

    args = parser.parse_args()

    differentiation_str = args.d[0]
    directory = args.p[0]+'/'+differentiation_str

    uti_sim_anneal = []
    uti_SAP = []
    uti_DRL_SP = []
    uti_MLP = []
    uti_ALL = []
    uti_DRL_HILL = []
    uti_HILL = []
    uti_DEFO = []

    cost_DEFO = []
    cost_SAP = []
    cost_DRL_SP = []
    cost_DRL_HILL = []
    cost_HILL = []

    filename_list = []
    X_axis = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    path_to_dir = "./Images/EVALUATION/"+differentiation_str+'/'

    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    aux_path_to_DEFO_results = "../DEFOResults/results-1-link_capacity-unif-05-1-zoo/"

    axis_counter = 0
    # Iterate over all topologies and evaluate our DRL agent on all TMs
    for filename in os.listdir(directory):
        dir_to_topology_rewards = directory+"/"+filename
        aux_sim_anneal = []
        aux_hill_climb = []
        aux_SAP = []
        aux_MLP = []
        aux_DRL_SP = []
        aux_DRL_HILL = []
        aux_DEFO = []

        aux_cost_SAP = []
        aux_cost_DEFO = []
        aux_cost_DRL_SP = []
        aux_cost_DRL_HILL = []
        aux_cost_hill_climb = []
        for subdir, dirs, files in os.walk(dir_to_topology_rewards):
            for file in files:
                if file.endswith((".pckl")):
                    top_name = file.split('.')[0]
                    tm_id = file.split('.')[1]
                    path_to_DEFO_file_results = aux_path_to_DEFO_results+top_name+"/standard_out_"+top_name+"_"+tm_id
                    defo_time = read_defo_optim_time(path_to_DEFO_file_results)

                    results = []
                    path_to_pckl_rewards = dir_to_topology_rewards + '/'

                    with open(path_to_pckl_rewards+file, 'rb') as f:
                        results = pickle.load(f)
                    
                    aux_sim_anneal.append(results[4])
                    aux_hill_climb.append(results[7])
                    aux_SAP.append(results[8])
                    aux_MLP.append(results[5])
                    aux_DRL_SP.append(results[9])
                    aux_DEFO.append(results[1])
                    aux_DRL_HILL.append(results[3])

                    aux_cost_SAP.append(results[13])
                    aux_cost_DEFO.append(defo_time)
                    aux_cost_DRL_SP.append(results[14])
                    aux_cost_DRL_HILL.append(results[16])
                    aux_cost_hill_climb.append(results[15])
                
        sim_anneal_mean = np.mean(aux_hill_climb) # np.mean(aux_sim_anneal)
        uti_SAP.append((sim_anneal_mean-np.mean(aux_SAP))/sim_anneal_mean)
        uti_MLP.append((sim_anneal_mean-np.mean(aux_MLP))/sim_anneal_mean)
        uti_DRL_SP.append((sim_anneal_mean-np.mean(aux_DRL_SP))/sim_anneal_mean)
        uti_DRL_HILL.append((sim_anneal_mean-np.mean(aux_DRL_HILL))/sim_anneal_mean)
        uti_HILL.append((sim_anneal_mean-np.mean(aux_hill_climb))/sim_anneal_mean)
        uti_DEFO.append((sim_anneal_mean-np.mean(aux_DEFO))/sim_anneal_mean)
        
        cost_DEFO.append(np.mean(aux_cost_DEFO))
        cost_SAP.append(np.mean(aux_cost_SAP))
        cost_DRL_SP.append(np.mean(aux_cost_DRL_SP))
        cost_DRL_HILL.append(np.mean(aux_cost_DRL_HILL))
        cost_HILL.append(np.mean(aux_cost_hill_climb))

        X_axis.append(axis_counter)
        filename_list.append(filename)
        axis_counter += 1

    # We do the following to order the scores
    for i in range(len(X_axis)):
        uti_ALL.append((filename_list[i], uti_SAP[i], uti_DRL_SP[i], uti_DRL_HILL[i]-uti_HILL[i], uti_DRL_HILL[i], cost_SAP[i], cost_DRL_SP[i], cost_DRL_HILL[i], uti_HILL[i], cost_HILL[i], uti_DEFO[i], cost_DEFO[i], 1, 0))

    new_uti_ALL = sorted(uti_ALL, key=lambda tup: tup[3], reverse=False)
    print(new_uti_ALL)
    print(len(new_uti_ALL))
    dict_tops = dict()
    dict_tops["BtAsiaPac"] = 1
    dict_tops["Goodnet"] = 1
    dict_tops["Garr199905"] = 1
    for i in range(len(X_axis)):
        if new_uti_ALL[i][0] in dict_tops:
            print(i, new_uti_ALL[i][0])
    uti_SAP = []
    uti_DRL_SP = []
    uti_DRL_HILL = []
    uti_HILL = []
    uti_DEFO = []

    cost_DEFO = []
    cost_SAP = []
    cost_DRL_SP = []
    cost_DRL_HILL = []
    cost_HILL = []
    for elem in new_uti_ALL:
        if elem[1]<-0.5:
            uti_SAP.append(-0.5)
        else:
            uti_SAP.append(elem[1])
        uti_DRL_SP.append(elem[2])
        uti_DRL_HILL.append(elem[4])
        uti_HILL.append(elem[8])
        uti_DEFO.append(elem[10])
        cost_SAP.append(elem[5])
        cost_DRL_SP.append(elem[6])
        cost_DRL_HILL.append(elem[7])
        cost_HILL.append(elem[9])
        cost_DEFO.append(elem[11])

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['pdf.fonttype'] = 42
    # Force ticks to appear
    plt.yticks(np.arange(-50, 60, 10))
    plt.ylim((-50, 60))

    plt.xticks(np.arange(0, 75, 8))
    plt.plot(X_axis, np.array(uti_SAP)*100, c='darkorange', linestyle='--', label="SAP", linewidth=3)
    #plt.plot(X_axis, uti_MLP, c='aqua', linestyle='-', label="MLP", linewidth=3)
    #plt.plot(X_axis, np.array(uti_DRL_SP)*100, c='dimgrey', linestyle='-.', label="DRL+GNN", linewidth=3)
    plt.plot(X_axis, np.array(uti_HILL)*100, c='darkgreen', linestyle='-', label="LS", linewidth=3)
    plt.plot(X_axis, np.array(uti_DEFO)*100, c='darkblue', linestyle=':', label="DEFO", linewidth=3)
    plt.plot(X_axis, np.array(uti_DRL_HILL)*100, c='red', linestyle='-.', label="Enero", linewidth=3)

    plt.ylabel("Relative performance w.r.t.\nLocal Search (%)", fontsize=16)
    plt.xlabel("Topology identifier", fontsize=19)
    #lgd = plt.legend(loc="lower left", bbox_to_anchor=(-0.3, -0.4), ncol=4)
    plt.grid(c='grey')
    plt.tight_layout()
    plt.savefig(path_to_dir+"Figure_9_Rel_Perf_all_tops.pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()
    plt.close()

    plt.rcParams.update({'font.size': 16})
    plt.rcParams['pdf.fonttype'] = 42
    # Force ticks to appear
    plt.yticks(np.arange(0, 200, 20))
    plt.ylim((0, 200))
    plt.xticks(np.arange(0, 75, 8))
    plt.plot(X_axis, np.array(cost_SAP), c='darkorange', linestyle='--', label="SAP", linewidth=2)
    plt.plot(X_axis, np.array(cost_HILL), c='darkgreen', linestyle='-', label="LS", linewidth=2)
    #plt.plot(X_axis, uti_MLP, c='aqua', linestyle='-', label="MLP", linewidth=2)
    #plt.plot(X_axis, np.array(cost_DRL_SP), c='dimgrey', linestyle='-.', label="DRL+GNN", linewidth=2)
    plt.plot(X_axis, np.array(cost_DEFO), c='darkblue', linestyle=':', label="DEFO", linewidth=2)
    plt.plot(X_axis, np.array(cost_DRL_HILL), c='red', linestyle='-.', label="Enero", linewidth=2)

    plt.ylabel("Average Execution Cost (s)", fontsize=16)
    plt.xlabel("Topology identifier", fontsize=19)
    plt.grid(c='grey')
    lgd = plt.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.33), ncol=4)
    plt.savefig(path_to_dir+"Cost_all_tops.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()
    n = np.arange(1,len(cost_SAP)+1) / np.float(len(cost_SAP))
    Xs = np.sort(cost_SAP)
    ax.step(Xs,n, c='darkorange', linestyle='--', label="SAP", linewidth=3) 
    Xs = np.sort(cost_HILL)
    ax.step(Xs,n,c='darkgreen', linestyle='-', label="LS", linewidth=3) 
    Xs = np.sort(cost_DEFO)
    ax.step(Xs,n,c='darkblue', linestyle=':', label="DEFO", linewidth=3) 
    Xs = np.sort(cost_DRL_HILL)
    ax.step(Xs,n,c='red', linestyle='-.', label="Enero", linewidth=3) 
    t = ax.text(50.0, 0.30, "Better", ha="center", va="center", size=15,
    bbox=dict(boxstyle="larrow,pad=0.2", fc="w", ec="k", lw=2))
    t = ax.text(165.0, 0.70, "DEFO", c="white", rotation=-45, ha="center", va="center", size=15,
    bbox=dict(boxstyle="rarrow,pad=0.2", fc="blue", ec="w", lw=2))

    plt.ylim((0, 1.005))
    plt.xlim((-2, 185.0))
    plt.xticks(np.arange(0, 185, 20))
    plt.ylabel('CDF', fontsize=17)
    plt.xlabel("Execution Cost (s)", fontsize=20)
    plt.legend(prop={'size': 12, 'weight': 'bold'}, loc='lower right')
    plt.grid(color='gray')
    lgd = plt.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.4), ncol=4)
    plt.tight_layout()
    #plt.show()
    plt.savefig(path_to_dir+'Figure_9_CDF_Cost_AllTops.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.clf()

    