import os
import subprocess
import argparse
from multiprocessing import Process
from multiprocessing import Pool, TimeoutError
import multiprocessing

def worker_execute(args):
    tm_id = args[0]
    model_id = args[1]
    drl_eval_res_folder = args[2]
    differentiation_str = args[3]
    graph_topology_name = args[4]
    dataset_folder = args[5]
    print("python script_eval_on_link_failure_topology.py -t "+str(tm_id)+" -m "+str(model_id)+" -g "+graph_topology_name+" -o "+drl_eval_res_folder+" -d "+differentiation_str+ ' -f ' + dataset_folder)
    subprocess.call(["python script_eval_on_link_failure_topology.py -t "+str(tm_id)+" -m "+str(model_id)+" -g "+graph_topology_name+" -o "+drl_eval_res_folder+" -d "+differentiation_str+ ' -f ' + dataset_folder], shell=True)

if __name__ == "__main__":
    # First we execute this script to evaluate our drl agent over different topologies from the topology zoo dataset
    # python eval_on_zoo_topologies.py -max_edge 100 -min_edge 2 -max_nodes 30 -min_nodes 1 -n 15 -d ./Logs/expSP_3top_15_B_NEWLogs.txt -f LinkFailure_EliBackbone
    
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='logs data file', type=str, required=True, nargs='+')
    parser.add_argument('-f', help='Link Failure dataset', type=str, required=True, nargs='+')
    parser.add_argument('-max_edge', help='maximum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_edge', help='minimum number of edges the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-max_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-min_nodes', help='minimum number of nodes the topology can have', type=int, required=True, nargs='+')
    parser.add_argument('-n', help='number of processes to use for the pool (number of DEFO instances running at the same time)', type=int, required=True, nargs='+')

    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    model_id = 0
    # Load best model
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break

    dataset_folder = "../Enero_datasets/dataset_sing_top/LinkFailure/"+args.f[0]+"/"
    # In this folder we store the rewards that later will be parsed for plotting
    drl_eval_res_folder = "../Enero_datasets/dataset_sing_top/LinkFailure/rwds-"+args.f[0]+"/"

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    if not os.path.exists(drl_eval_res_folder):
        os.makedirs(drl_eval_res_folder)

    if not os.path.exists(drl_eval_res_folder+differentiation_str):
        os.makedirs(drl_eval_res_folder+differentiation_str)
    else:
        os.system("rm -rf %s" % (drl_eval_res_folder+differentiation_str))
        os.makedirs(drl_eval_res_folder+differentiation_str)

    # Iterate over all topologies and evaluate our DRL agent on all TMs
    for subdir, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith((".graph")):
                topology_dir = dataset_folder + file.split('.')[0] + '/'
                topology_num_nodes = 0
                with open(topology_dir+file) as fd:
                    # Loop to read the Number of NODES and EDGES
                    while (True):
                        line = fd.readline()
                        if (line == ""):
                            break
                        if (line.startswith("NODES")):
                            topology_num_nodes = int(line.split(' ')[1])

                        # If we are inside the range of number of nodes
                        if topology_num_nodes>=args.min_nodes[0] and topology_num_nodes<=args.max_nodes[0]:
                            if (line.startswith("EDGES")):
                                topology_num_edges = int(line.split(' ')[1])
                                # If we are inside the range of number of edges
                                if topology_num_edges<=args.max_edge[0] and topology_num_edges>=args.min_edge[0]:
                                    topology_Name = file.split('.')[0]
                                    print("*****")
                                    print("***** Evaluating on file: "+file+" with number of edges "+str(topology_num_edges))
                                    print("*****")
                                    my_dataset_folder = dataset_folder+topology_Name
                                    argums = [(tm_id, model_id, drl_eval_res_folder, differentiation_str, topology_Name, my_dataset_folder) for tm_id in range(15)]
                                    with Pool(processes=args.n[0]) as pool:
                                        pool.map(worker_execute, argums)
                        else:
                            break
