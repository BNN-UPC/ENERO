import os
import subprocess
import numpy as np
import argparse
import time
from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
import multiprocessing
import argparse

# This script must be executed from the REPETITA folder! 
# See https://github.com/svissicchio/Repetita for more details
# First we must execute the run_Repetita_all_Topologies.py to store the DEFO results and then
# we can use the generated dataset to train a DRL agent and to evaluate the trained model on all topologies

repetita_folder_name = "results-1-link_capacity-unif-05-1-zoo"
dataset_folder = "../Enero_datasets/"+repetita_folder_name+"/"
results_folder = "../DEFOResults/"

def worker_execute(args):
    tm_id = args[0]
    fileName = args[1]
    dataset_folder = args[2]+fileName+'/'
    results_folder = args[3]+fileName+'/'
    optim_time = args[4]

    fileName_demand = dataset_folder+'TM/'+fileName+'.'+str(tm_id)+'.demands'
    fileName_graph = dataset_folder+fileName+'.graph'
    fileName_output = results_folder+'res_'+fileName+'_'+str(tm_id)
    fileName_standrd_output = results_folder+'standard_out_'+fileName+'_'+str(tm_id)
    subprocess.call(["./repetita -graph "+str(fileName_graph)+" -demands "+str(fileName_demand)+" -solver defoCP -t "+str(optim_time)+" -scenario SingleSolverRun -outpaths "+str(fileName_output)+" -out "+str(fileName_standrd_output)+" -verbose 1 >> "+str(fileName_standrd_output)], shell=True)

def play_defo(min_nodes, max_nodes, min_edges, max_edges, optim_time, num_procc, optimizer):
    list_optimizers = []
    if (optimizer[0]=="1"):
        list_optimizers.append("defoCP")
    if (optimizer[1]=="1"):
        list_optimizers.append("SRLS")
    if (optimizer[2]=="1"):
        list_optimizers.append("TabuIGPWO")
    
    topology_num_nodes = 50
    first_time = True # We use this flag to remove the directory with old data on the first acces 
    for subdir, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith((".graph")):
                fileName = file.split('.')[0]
                with open(dataset_folder+fileName+'/'+file) as fd:
                    while (True):
                        line = fd.readline()
                        if (line == ""):
                            break
                        if (line.startswith("NODES")):
                            topology_num_nodes = int(line.split(' ')[1])
                        if (line.startswith("EDGES")) and topology_num_nodes<=max_nodes and topology_num_nodes>=min_nodes:
                            topology_num_edges = int(line.split(' ')[1])
                            double_link = 0 # Indicates if we have found a double link in the same direction to skip the topology
                            dict_double_link = dict()
                            for line in fd:
                                if (not line.startswith("Link_") and not line.startswith("edge_")):
                                    continue
                                camps = line.split(" ")
                                # We want to exclude the topologies that have multiple links between two nodes: (0,1), (0,1), (1,0)
                                if camps[1]+':'+camps[2] in dict_double_link:
                                    double_link = 1
                                    print("Double link for ", file)
                                    break
                                else:
                                    dict_double_link[camps[1]+':'+camps[2]] = 1
                            # We just evaluate our model in a subset number of topologies
                            if double_link<=0 and topology_num_edges>=min_edges and topology_num_edges<=max_edges:      
                                for optimizer in list_optimizers:
                                    print("***** Generating "+optimizer+" solution on file: "+file+" with number of edges "+str(topology_num_edges)+" and number of nodes "+str(topology_num_nodes))
                                    #results_folder_optimizer = results_folder+optimizer+"/"+repetita_folder_name+"_"+str(min_edges)+"_"+str(max_edges)+"/"
                                    results_folder_optimizer = results_folder+optimizer+"/"+repetita_folder_name+"/"

                                    if not os.path.exists(results_folder_optimizer):
                                        os.makedirs(results_folder_optimizer) 
                                    elif first_time:
                                        os.system("rm -rf %s" % (results_folder_optimizer))
                                        os.makedirs(results_folder_optimizer)

                                    config_file = results_folder_optimizer+"CONFIG_REPETITA.txt"
                                    with open(config_file, 'w') as fd2:
                                        fd2.write('-max_edges '+str(max_edges)+'\n')
                                        fd2.write('-min_edges '+str(min_edges)+'\n')
                                        fd2.write('-max_nodes '+str(max_nodes)+'\n')
                                        fd2.write('-min_nodes '+str(min_nodes)+'\n')
                                        fd2.write('-optim_time '+str(optim_time)+'\n')
                                        fd2.write('-n '+str(num_procc)+'\n')
                                    
                                    if not os.path.exists(results_folder_optimizer+fileName):
                                        os.makedirs(results_folder_optimizer+fileName) 
                                    elif first_time:
                                        os.system("rm -rf %s" % (results_folder_optimizer+fileName))
                                        os.makedirs(results_folder_optimizer+fileName)
                                    # We copy the TMs and the .graph files
                                    #os.system("cp -r %s/*%s* %s" % (dataset_folder, fileName, results_folder_optimizer))
                                    #os.system("cp -r %sCONFIGURATION.txt %s" % (dataset_folder, results_folder_optimizer))
                                    args = [(tm_id, fileName, dataset_folder, results_folder_optimizer, optim_time) for tm_id in range(15)]
                                    with Pool(processes=num_procc) as pool:
                                        pool.map(worker_execute, args)
                                
                                first_time = False

if __name__ == "__main__":
    # python3 run_Defo_all_Topologies.py -max_edges 80 -min_edges 20 -max_nodes 25 -min_nodes 5 -optim_time 10 -n 15 --optimizer 100
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-max_edges', help='maximum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_edges', help='minimum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-max_nodes', help='maximum number of nodes the topology can have', type=int, required=False, nargs='+', default=100)
    parser.add_argument('-min_nodes', help='minimum number of nodes the topology can have', type=int, required=False, nargs='+', default=0)
    parser.add_argument('-optim_time', help='optimization time used for DEFO', type=int, required=True, nargs='+')
    parser.add_argument('-n', help='number of processes to use for the pool (number of DEFO instances running at the same time)', type=int, required=True, nargs='+')
    parser.add_argument("--optimizer", help="defoCP(100) SRLS(010) TabuIGPWO(001)",default="111")

    args = parser.parse_args()

    if (len(args.optimizer) != 3):
        print ("ERROR: Store parameter should contain only 3 characters")
        exit()
    for c in args.optimizer:
        if (c != "0" and c != "1"):
            print ("ERROR: Store parameter should contain only 3 characters with 0 or 1")
            exit()
    #Iterate over all '.graph' files and for each of them iterate over all TMs and execute repetita to store the results in the proper folder
    play_defo(args.min_nodes[0], args.max_nodes[0], args.min_edges[0], args.max_edges[0], args.optim_time[0], args.n[0], args.optimizer)
