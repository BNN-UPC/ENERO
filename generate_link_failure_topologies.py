import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
import networkx as nx

SEED = 7
random.seed(SEED)
link_capacities = [45000, 45000]

def process_graph_file(graph_file):
    Gbase = nx.DiGraph()
    dict_links = dict()
    net_size = 0

    with open(graph_file) as fd:
        line = fd.readline()
        camps = line.split(" ")
        net_size = int(camps[1])
        # Remove : label x y
        line = fd.readline()
        
        for i in range (net_size):
            line = fd.readline()
            node = line[0:line.find(" ")]
            
        for line in fd:
            if (not line.startswith("Link_") and not line.startswith("edge_")):
                continue
            camps = line.split(" ")
            src = int(camps[1])
            dst = int(camps[2])
            bw = int(camps[4])
            dict_links[str(src)+':'+str(dst)] = bw
            Gbase.add_edge(src, dst)
    
    return Gbase, dict_links, net_size

def write_graph_to_file(G_copy, dict_links, num_nodes, graph_file, top):
    num_edges = len(G_copy.edges())
    with open(graph_file, 'a') as fd:
        fd.write('NODES '+str(num_nodes)+'\n')
        fd.write('label x y\n')
        for n in range(num_nodes):
            fd.write(str(n)+'_Node '+str(0.0)+' '+str(0.0)+'\n')
        
        fd.write('\n')
        fd.write('EDGES '+str(num_edges)+'\n')
        fd.write('label src dest weight bw delay\n')
        edge_iter = 0
        for i in G_copy:
            for j in G_copy[i]:
                bw = dict_links[str(i)+':'+str(j)]
                fd.write('edge_'+str(edge_iter)+' '+str(i)+' '+str(j)+' '+str(1)+' '+str(bw)+' '+str(100)+'\n')
                edge_iter += 1

def get_traffic_matrix(graph_file, traffic_file):
    net_size = 0
    with open(graph_file) as fd:
        line = fd.readline()
        net_size = int(line.split(' ')[1])
    
    tm = np.zeros((net_size,net_size))
    with open(traffic_file) as fd:
        fd.readline()
        fd.readline()
        for line in fd:
            camps = line.split(" ")
            tm[int(camps[1]),int(camps[2])] = float(camps[3])
    return (net_size, tm)

def compute_new_TM(net_size, original_TM):
    new_tm = np.zeros((net_size,net_size))
    for src in range(net_size):
        for dst in range(net_size):
            if src!=dst:
                noise = np.random.normal(loc=0.0, scale=200, size=1)
                # To make sure we don't have negative bw
                while np.floor(noise+original_TM[src,dst])<=0:
                    noise = np.random.normal(loc=0.0, scale=200, size=1)
                new_tm[src, dst] = np.floor(noise+original_TM[src,dst])

    return new_tm

if __name__ == "__main__":
    # python3 generate_link_failure_topologies.py -d results-1-link_capacity-unif-05-1-zoo -topology HurricaneElectric -num_topologies 1 -link_failures 1
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='Directory where I can find the topology', type=str, required=True, nargs='+')
    parser.add_argument('-topology', help='Name of the topology from TopologyZoo that we want to replicate', type=str, required=True, nargs='+')
    parser.add_argument('-num_topologies', help='How many NEW topologies we want', type=int, required=True, nargs='+')
    parser.add_argument('-link_failures', help='NUM links we want to DELETE', type=int, required=True, nargs='+')

    args = parser.parse_args()

    dataset_folder_name = "../Enero_datasets/"+args.d[0]+"/results_zoo/"
    graph_topology_name = args.topology[0]
    num_link_failures = args.link_failures[0]
    num_topologies_x_link_failure = args.num_topologies[0]
    
    dir_new_synth_dataset = "../Enero_datasets/dataset_sing_top/LinkFailure/LinkFailure_"+graph_topology_name+"/"
    if not os.path.exists(dir_new_synth_dataset):
        os.makedirs(dir_new_synth_dataset)
    else:
        os.system("rm -rf %s" % (dir_new_synth_dataset))
        os.makedirs(dir_new_synth_dataset)

    config_file = dir_new_synth_dataset+"CONFIG_LINK_FAILURE.txt"
    with open(config_file, 'w') as fd2:
        fd2.write('Topology Name: '+str(graph_topology_name)+'\n')
        fd2.write('MAX Number of Link Failures: '+str(num_link_failures)+'\n')
        fd2.write('num_topologies_x_link_failure: '+str(num_topologies_x_link_failure)+'\n')
    
    # Remove existing topologies
    os.system("rm -rf %s/%s*" % (dir_new_synth_dataset, graph_topology_name))
    
    graph_file = dataset_folder_name+"/"+graph_topology_name+"/"+graph_topology_name+".graph"
    Gbase, original_dict_links, num_nodes = process_graph_file(graph_file)
    print("Original topology number of edges: ", len(Gbase.edges()))

    link_failure = 1
    while link_failure<=num_link_failures:
        store_link_combinations = dict()
        for top in range(num_topologies_x_link_failure):
            while True:
                print("Generating topology with ", link_failure, " link failures")
                removed_links = 0
                G_copy = Gbase.copy()
                dict_links = original_dict_links.copy()

                # Remove original links if 
                while removed_links<link_failure:
                    edge = random.choice(list(dict_links.items()))
                    e0 = int(edge[0].split(':')[0])
                    e1 = int(edge[0].split(':')[1])
                    G_copy.remove_edge(e0, e1)
                    G_copy.remove_edge(e1, e0)
                    del dict_links[str(e0)+':'+str(e1)]
                    del dict_links[str(e1)+':'+str(e0)]
                    removed_links += 1
                
                # Iterate over edges in a deterministic way and make sure this combination of edges is unique
                # This is done to avoid having multiple graphs with the same edges
                key_edges = ""
                for i in G_copy:
                    for j in G_copy[i]:
                        key_edges += ":"+str(i)+'-'+str(j)

                # We just consider valid the topology if it's connected and if we didn't generate a similar topology before
                if nx.is_connected(G_copy.to_undirected()) and key_edges not in store_link_combinations:
                    store_link_combinations[key_edges] = 1
                    print("Topology "+str(top)+" with number of edges: "+str(len(G_copy.edges()))+'/'+str(len(Gbase.edges())))
                    # nx.draw(G_copy, with_labels=True)
                    # plt.show()
                    dir_new_topology_with_failure = dir_new_synth_dataset+graph_topology_name+"_"+str(link_failure)+"_"+str(top)
                    if not os.path.exists(dir_new_topology_with_failure):
                        os.makedirs(dir_new_topology_with_failure) 
                    else:
                        os.system("rm -rf %s" % (dir_new_topology_with_failure))
                        os.makedirs(dir_new_topology_with_failure)
                    
                    # Create folder to copy the original TMs
                    os.makedirs(dir_new_topology_with_failure+'/TM/')

                    # We copy the TMs and the .graph files
                    os.system("cp -r %s/%s/TM/ %s/" % (dataset_folder_name, graph_topology_name, dir_new_topology_with_failure))

                    for tm_id in range(150):
                        os.system("mv %s/TM/%s.%s.demands %s/TM/%s.%s.demands" % (dir_new_topology_with_failure, graph_topology_name, str(tm_id), dir_new_topology_with_failure, graph_topology_name+"_"+str(link_failure)+"_"+str(top), str(tm_id)))

                    new_graph_file = dir_new_topology_with_failure+"/"+graph_topology_name+"_"+str(link_failure)+"_"+str(top)+'.graph'
                    write_graph_to_file(G_copy, dict_links, num_nodes, new_graph_file, top)

                    break
        link_failure += 1