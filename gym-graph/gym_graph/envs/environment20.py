import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import os.path
import json 
import gc
import defo_process_results as defoResults

class Env20(gym.Env):
    """
    Similar to environment15.py but this one is used for the SAP (instead of hill climbing)

    Environment used in the middlepoint routing problem.
    We are using bidirectional links in this environment!
    In this environment we make the MP between nodes and concatenate the edge features
    to the node features in the message function.
    self.edge_state[:][0] = link utilization
    self.edge_state[:][1] = link capacity
    self.edge_state[:][2] = bw allocated (the one that goes from src to dst)
    """
    def __init__(self):
        self.graph = None # Here we store the graph as DiGraph (without repeated edges)
        self.source = None
        self.destination = None
        self.demand = None

        self.edge_state = None
        self.graph_topology_name = None # Here we store the name of the graph topology from the repetita dataset
        self.dataset_folder_name = None # Here we store the name of the repetita dataset being used: 2015Defo, 2016TopologyZoo_unary,2016TopologyZoo_inverseCapacity, etc. 

        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.between_feature = None

        self.nodeId = None
        self.sp_middlepoints = None # For each src,dst we store the nodeId of the sp middlepoint
        self.shortest_paths = None # For each src,dst we store the shortest path to reach d

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        # Episode length in timesteps
        self.episode_length = None
        self.list_eligible_demands = None # Here we store those demands from DEFO that have one middlepoint. These demands are going to be eligible by our DRL agent.
        self.iter_list_elig_demn = None

        # Error at the end of episode to evaluate the learning process
        self.error_evaluation = None
        # Ideal target link capacity: self.sumTM/self.numEdges
        self.target_link_capacity = None

        self.TM = None # Traffic matrix where self.TM[src][dst] indicates how many packets are sent from src to dst
        self.meanTM = None
        self.stdTM = None
        self.sumTM = None
        self.routing = None # Loaded routing matrix
        self.paths_Matrix_from_routing = None # We store a list of paths extracted from the routing matrix for each src-dst pair

        self.K = None
        self.nodes = None # List of nodes to pick randomly from them
        self.ordered_edges = None
        self.edgesDict = dict() # Stores the position id of each edge in order
        self.previous_path = None

        self.src_dst_k_middlepoints = None # For each src, dst, we store the k middlepoints
        self.node_to_index_dic = None # For each node from the real graph we store it's index
        self.index_to_node_lst = None # We store a list of nodes in an ordered fashion

        self.numNodes = None
        self.numEdges = None
        self.numSteps = 0 # As our problem can go forever, we limit it to 10 steps

        self.sameLink = False # Indicates if we are working with the same link

        # We store the edge that has maximum utilization
        # (src, dst, MaxUtilization)
        self.edgeMaxUti = None 
        # We store the edge that has minimum utilization
        # (src, dst, MaxUtilization)
        self.edgeMinUti = None 
        # We store the path with more bandwidth from the edge with maximum utilization
        # (src, dst, MaxBandwidth)
        self.patMaxBandwth = None 
        self.maxBandwidth = None

        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() # Stores the paths for each src:dst pair

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            for adj in self.graph[node]:
                if not 'betweenness' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['betweenness'] = 0
                if not 'edgeId' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['edgeId'] = incId
                if not 'numsp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['numsp'] = 0
                if not 'utilization' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['utilization'] = 0
                if not 'capacity' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['capacity'] = 0
                if not 'weight' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['weight'] = 0
                if not 'kshortp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['kshortp'] = 0
                if not 'crossing_paths' in self.graph[node][adj][0]: # We store all the src,dst from the paths crossing each edge
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1

    def _generate_tm(self, tm_id):
        # Sample a file randomly to initialize the tm
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        self.TM = self.defoDatasetAPI._get_traffic_matrix(tm_file)

        self.iter_list_elig_demn = 0
        self.list_eligible_demands.clear()
        min_links_bw = 1000000.0
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))
                    # If we have a link between src and dst
                    if src in self.graph and dst in self.graph[src]:
                        # Store the link with minimum bw
                        if self.links_bw[src][dst]<min_links_bw:
                            min_links_bw = self.links_bw[src][dst]
                        
                        # Clear the link utilization and crossing paths for each link
                        self.graph[src][dst][0]['utilization'] = 0.0
                        self.graph[src][dst][0]['crossing_paths'].clear()

        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)

        self.maxBandwidth = np.amax(self.TM)

        traffic = np.copy(self.TM)
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        self.sumTM = np.sum(traffic)
        self.target_link_capacity = self.sumTM/self.numEdges
        self.meanTM = np.mean(traffic)
        self.stdTM = np.std(traffic)

    def compute_SPs(self):
        diameter = nx.diameter(self.graph)
        sp_path = self.dataset_folder_name+"/K_shortest_paths.json"

        if not os.path.isfile(sp_path):
            for n1 in self.graph:
                for n2 in self.graph:
                    if (n1 != n2):
                        # Check if we added the element of the matrix
                        if str(n1)+':'+str(n2) not in self.allPaths:
                            self.allPaths[str(n1)+':'+str(n2)] = []

                        # First we compute the shortest paths taking into account the diameter
                        [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=diameter*2)]

                        # We take all the paths from n1 to n2 and we order them according to the path length
                        # sorted() ordena los paths de menor a mayor numero de
                        # saltos y los que tienen los mismos saltos te los ordena por indice
                        self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

                        path = 0
                        while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                            path = path + 1

                        # Remove paths not needed
                        del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                        gc.collect()
            
            with open(sp_path, 'w') as fp:
                json.dump(self.allPaths, fp)
        else:
            self.allPaths = json.load(open(sp_path))

    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K):
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        self.list_eligible_demands = list()
        self.iter_list_elig_demn = 0

        self.maxCapacity = 0 # We take the maximum capacity to normalize

        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        
        self.node_to_index_dic = self.defoDatasetAPI.node_to_index_dic_pvt
        self.index_to_node_lst = self.defoDatasetAPI.index_to_node_lst_pvt

        self.graph = self.defoDatasetAPI.Gbase
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.K = 5 # We try to allocate on the 10 Shortest Paths

        self.edge_state = np.zeros((self.numEdges, 3))

        position = 0
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.defoDatasetAPI.links_bw[i][j]
                self.graph[i][j][0]['weight'] = self.defoDatasetAPI.links_weight[i][j]
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))
        self.compute_SPs()

    def step(self, action, demand, source, destination):
        # Action is the middlepoint. Careful because it can also be action==destination if src,dst are connected directly by an edge
        self.reward = 0
        self.episode_over = False

        pathList = self.allPaths[str(source) +':'+ str(destination)]
        currentPath = pathList[0]

        # If we can allocate it somewhere and the uti doesn't pass the link capacity
        if action!=-1:
            currentPath = pathList[action]
        # If we can't allocate the action, we perform load balancing
        else: 
            action = random.randint(0, len(pathList)-1)
            currentPath = pathList[action]

        i = 0
        j = 1

        # 2. Iterate over pairs of nodes and allocate the demand
        while j < len(currentPath):
            self.graph[currentPath[i]][currentPath[j]][0]['utilization'] += demand
            i = i + 1
            j = j + 1
        
        # Find new maximum and minimum utilization link
        maxUti = 0
        minUti = 1000000
        self.error_evaluation = 0
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                if self.edge_state[position][0]/link_capacity>maxUti:
                    maxUti = self.edge_state[position][0]/link_capacity
                    self.edgeMaxUti = (i, j, maxUti)
                if self.edge_state[position][0]/link_capacity<minUti:
                    minUti = self.edge_state[position][0]/link_capacity
                    self.edgeMinUti = (i, j, minUti)
                self.error_evaluation = self.error_evaluation + (self.target_link_capacity - self.edge_state[position][0])
        
        if self.iter_list_elig_demn<len(self.list_eligible_demands):
            self._obtain_demand()
        else:
            src = 1
            dst = 2
            bw = self.TM[src][dst]
            self.patMaxBandwth = (src, dst, int(bw))
            self.episode_over = True

        return self.episode_over, np.absolute(self.error_evaluation), self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1], self.edgeMaxUti, self.edgeMinUti[2], np.std(self.edge_state[:,0])

    def _obtain_demand(self):
        src = self.list_eligible_demands[self.iter_list_elig_demn][0]
        dst = self.list_eligible_demands[self.iter_list_elig_demn][1]
        bw = self.list_eligible_demands[self.iter_list_elig_demn][2]
        self.patMaxBandwth = (src, dst, int(bw))
        self.iter_list_elig_demn += 1

    def reset(self, tm_id):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        self._generate_tm(tm_id)

        # Clear the link utilization and crossing paths
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
            
        self._obtain_demand()

        return self.TM[self.patMaxBandwth[0]][self.patMaxBandwth[1]], self.patMaxBandwth[0], self.patMaxBandwth[1]