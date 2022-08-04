import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import json 
import os.path
import gc
import defo_process_results as defoResults

class Env15(gym.Env):
    """
    Environment used for the simulated annealing and hill climbing benchmarks in the 
    script_eval_on_single_topology.py with SP only! No ecmp at all here!

    Environment used in the middlepoint routing problem using SP to reach a middlepoint.
    We are using bidirectional links in this environment!
    self.edge_state[:][0] = link utilization
    self.edge_state[:][1] = link capacity
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
        self.list_of_demands_to_change = None # Eligible demands coming from the DRL agent

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.between_feature = None

        self.sp_middlepoints = None # For each src,dst we store the nodeId of the sp middlepoint
        self.shortest_paths = None # For each src,dst we store the shortest path to reach d

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        # Episode length in timesteps
        self.episode_length = None

        self.list_eligible_demands = None # Here we store those demands from DEFO that have one middlepoint. These demands are going to be eligible by our DRL agent.
        self.num_critical_links = 5

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
                if not 'crossing_paths' in self.graph[node][adj][0]: # We store all the src,dst from the paths crossing each edge
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1
    
    def decrease_links_utilization_sp(self, src, dst, init_source, final_destination):
        # In this function we desallocate the bandwidth by segments. This funcion is used when we want
        # to desallocate from a src to a middlepoint and then from middlepoint to a dst using the sp

        # We obtain the demand from the original source,destination pair
        bw_allocated = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]

        i = 0
        j = 1
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] -= bw_allocated 
            if str(init_source)+':'+str(final_destination) in self.graph[firstNode][secondNode][0]['crossing_paths']:
                del self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)]
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1

    def _generate_tm(self, tm_id):        
        # Sample a file randomly to initialize the tm
        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        # This 'results_file' file is ignored!
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_"+str(tm_id)
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+'.'+str(tm_id)+".demands"
        
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)
        self.links_bw = self.defoDatasetAPI.links_bw
        self.TM = self.defoDatasetAPI._get_traffic_matrix(tm_file)

        self.maxBandwidth = np.amax(self.TM)

        traffic = np.copy(self.TM)
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        self.sumTM = np.sum(traffic)
        self.target_link_capacity = self.sumTM/self.numEdges
        self.meanTM = np.mean(traffic)
        self.stdTM = np.std(traffic)
    
    def compute_link_utilization_reset_sp(self):
        # Compute the paths that cross each link and then add up the bandwidth to obtain the link utilization
        for src in range (0,self.numNodes):
            for dst in range (0,self.numNodes):
                if src!=dst:
                    self.allocate_to_destination_sp(src, dst, src, dst)

    def mark_edges(self, action_flags, src, dst, init_source, final_destination):
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            action_flags[self.edgesDict[str(firstNode)+':'+str(secondNode)]] += 1.0
            i = i + 1
            j = j + 1

    
    def mark_action_to_edges(self, first_node, init_source, final_destination): 
        # In this function we mark for each link which is the bw that it will allocate. This we will
        # use to avoid repeated actions
        action_flags = np.zeros(self.numEdges)
        
        # Mark until first_node
        self.mark_edges(action_flags, init_source, first_node, init_source, final_destination)

        # If the first node is a middlepoint
        if first_node!=final_destination:
            self.mark_edges(action_flags, first_node, final_destination, init_source, final_destination)
        
        return action_flags

    def compute_middlepoint_set_remove_rep_actions_no_loop(self):
        # In this function we compute the middlepoint set but we don't take into account the middlepoints whose 
        # actions are repeated and neither those middlepoints whose SPs pass over the DST node
        
        # Compute SPs for each src,dst pair
        self.compute_SPs()

        # We compute the middlepoint set for each src,dst pair and we don't consider repeated actions
        self.src_dst_k_middlepoints = dict()
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)] = list()
                    repeated_actions = list()
                    for midd in range (0,self.K):
                        # If the middlepoint is not the source node
                        if midd!=n1:
                            action_flags = self.mark_action_to_edges(midd, n1, n2)
                            # If we allocated to a middlepoint that is not the final destination
                            if midd!=n2:
                                # If the repeated_actions list is empty we make the following verifications
                                if len(repeated_actions) == 0:

                                    path1 = self.shortest_paths[n1, midd]
                                    path2 = self.shortest_paths[midd, n2]

                                    # Check that the dst node is not in the SP to avoid loops!
                                    currentPath = path1[:len(path1)-1]+path2
                                    dst_counter = 0
                                    for node in currentPath:
                                        if node==n2 or node==n1:
                                            dst_counter += 1
                                    # If there is only one dst node
                                    if dst_counter==2:
                                        repeated_actions.append(action_flags)
                                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                else:
                                    repeatedAction = False
                                    # Compare the current action with the previous ones
                                    for previous_actions in repeated_actions:
                                        subtraction = np.absolute(np.subtract(action_flags,previous_actions))
                                        if np.sum(subtraction)==0.0:
                                            repeatedAction = True
                                            break
                                    # If we didn't find any identical action, we make the following verifications
                                    if not repeatedAction:                                        
                                        path1 = self.shortest_paths[n1, midd]
                                        path2 = self.shortest_paths[midd, n2]
                                        # Check that the dst node is not in the SP to avoid loops!
                                        currentPath = path1[:len(path1)-1]+path2
                                        dst_counter = 0
                                        for node in currentPath:
                                            if node==n2 or node==n1:
                                                dst_counter += 1
                                        # If there is only one dst node
                                        if dst_counter==2:
                                            self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                            repeated_actions.append(action_flags)

                            else: 
                                # If it's the first action we add it to the repeated actions list
                                if len(repeated_actions) == 0:
                                    self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                    repeated_actions.append(action_flags)
                                else:
                                    repeatedAction = False
                                    # Compare the current action with the previous ones
                                    for previous_actions in repeated_actions:
                                        subtraction = np.absolute(np.subtract(action_flags,previous_actions))
                                        if np.sum(subtraction)==0.0:
                                            repeatedAction = True
                                            break
                                    
                                    # If we didn't find any identical action, we add the middlepoint to the set
                                    if not repeatedAction:
                                        self.src_dst_k_middlepoints[str(n1)+':'+str(n2)].append(midd)
                                        repeated_actions.append(action_flags)

    def compute_SPs(self):
        diameter = nx.diameter(self.graph)
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype=object)
        
        allPaths = dict()
        sp_path = self.dataset_folder_name+"/shortest_paths.json"

        if not os.path.isfile(sp_path):
            for n1 in range (0,self.numNodes):
                for n2 in range (0,self.numNodes):
                    if (n1 != n2):
                        allPaths[str(n1)+':'+str(n2)] = []
                        # First we compute the shortest paths taking into account the diameter
                        [allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=diameter*2)]                    # We take all the paths from n1 to n2 and we order them according to the path length
                        # sorted() ordena los paths de menor a mayor numero de
                        # saltos y los que tienen los mismos saltos te los ordena por indice
                        aux_sorted_paths = sorted(allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))                    # self.shortest_paths[n1,n2] = nx.shortest_path(self.graph, n1, n2,weight='weight')
                        allPaths[str(n1)+':'+str(n2)] = aux_sorted_paths[0]
        
            with open(sp_path, 'w') as fp:
                json.dump(allPaths, fp)
        else:
            allPaths = json.load(open(sp_path))

        for n1 in range (0,self.numNodes):
            for n2 in range (0,self.numNodes):
                if (n1 != n2):
                    self.shortest_paths[n1,n2] = allPaths[str(n1)+':'+str(n2)]
        
    def generate_environment(self, dataset_folder_name, graph_topology_name, EPISODE_LENGTH, K, percentage_demands):
        self.episode_length = EPISODE_LENGTH
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        self.list_eligible_demands = list()
        self.percentage_demands = percentage_demands

        self.maxCapacity = 0 # We take the maximum capacity to normalize

        graph_file = self.dataset_folder_name+"/"+self.graph_topology_name+".graph"
        results_file = self.dataset_folder_name+"/res_"+self.graph_topology_name+"_0"
        tm_file = self.dataset_folder_name+"/TM/"+self.graph_topology_name+".0.demands"
        self.defoDatasetAPI = defoResults.Defo_results(graph_file,results_file)

        self.node_to_index_dic = self.defoDatasetAPI.node_to_index_dic_pvt
        self.index_to_node_lst = self.defoDatasetAPI.index_to_node_lst_pvt

        self.graph = self.defoDatasetAPI.Gbase
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.K = K
        if self.K>self.numNodes:
            self.K = self.numNodes

        self.edge_state = np.zeros((self.numEdges, 2))
        self.shortest_paths = np.zeros((self.numNodes,self.numNodes),dtype="object")

        position = 0
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.defoDatasetAPI.links_bw[i][j]
                self.graph[i][j][0]['weight'] = self.defoDatasetAPI.links_weight[i][j]
                if self.graph[i][j][0]['capacity']>self.maxCapacity:
                    self.maxCapacity = self.graph[i][j][0]['capacity']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
                position += 1

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))

        self.compute_middlepoint_set_remove_rep_actions_no_loop()
    
    def step_sp(self, action, source, destination):
        # We get the K-middlepoints between source-destination
        middlePointList = list(self.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint using the shortest path
        self.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination using the shortest path
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        # Find new maximum and minimum utilization link
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)

        return self.edgeMaxUti[2]
    
    def step_hill_sp(self, action, source, destination):
        # We get the K-middlepoints between source-destination
        middlePointList = list(self.src_dst_k_middlepoints[str(source) +':'+ str(destination)])
        middlePoint = middlePointList[action]

        # First we allocate until the middlepoint using the shortest path
        self.allocate_to_destination_sp(source, middlePoint, source, destination)
        # If we allocated to a middlepoint that is not the final destination
        if middlePoint!=destination:
            # Then we allocate from the middlepoint to the destination using the shortest path
            self.allocate_to_destination_sp(middlePoint, destination, source, destination)
            # We store that the pair source,destination has a middlepoint
            self.sp_middlepoints[str(source)+':'+str(destination)] = middlePoint
        
        # Find new maximum and minimum utilization link
        old_Utilization = self.edgeMaxUti[2]
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)

        return -self.edgeMaxUti[2]
    
    def reset_sp(self, tm_id):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        self._generate_tm(tm_id)

        self.sp_middlepoints = dict()

        # Clear the link utilization and crossing paths
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()

        # For each link we store the total sum of bandwidths of the paths crossing each link without middlepoints
        self.compute_link_utilization_reset_sp()

        # We iterate over all links in an ordered fashion and store the features to edge_state
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        return self.edgeMaxUti[2]

    def reset_hill_sp(self, tm_id):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        self._generate_tm(tm_id)

        self.sp_middlepoints = dict()

        # Clear the link utilization and crossing paths
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()

        # For each link we store the total sum of bandwidths of the paths crossing each link without middlepoints
        self.compute_link_utilization_reset_sp()

        # We iterate over all links in an ordered fashion and store the features to edge_state
        self.edgeMaxUti = (0, 0, 0)
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        return -self.edgeMaxUti[2]

    def _get_top_k_critical_flows(self, list_ids):
        self.list_eligible_demands.clear()
        for linkId in list_ids:
            i = linkId[0]
            j = linkId[1]
            for demand, value in self.graph[i][j][0]['crossing_paths'].items():
                src, dst = int(demand.split(':')[0]), int(demand.split(':')[1])
                if (src, dst, self.TM[src,dst]) not in self.list_eligible_demands:  
                    self.list_eligible_demands.append((src, dst, self.TM[src,dst]))

        self.list_eligible_demands = sorted(self.list_eligible_demands, key=lambda tup: tup[2], reverse=True)
        if len(self.list_eligible_demands)>int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands)):
            self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]

    def reset_DRL_hill_sp(self, tm_id, best_routing, list_of_demands_to_change):
        """
        Reset environment and setup for new episode. 
        Generate new TM but load the same routing. We remove the path with more bandwidth
        from the link with more utilization to later allocate it on a new path in the act().
        """
        self._generate_tm(tm_id)
        if best_routing is not None:
            self.sp_middlepoints = best_routing
        else: 
            self.sp_middlepoints = dict()
        self.list_of_demands_to_change = list_of_demands_to_change

        # Clear the link utilization and crossing paths
        for i in self.graph:
            for j in self.graph[i]:
                self.graph[i][j][0]['utilization'] = 0.0
                self.graph[i][j][0]['crossing_paths'].clear()
        
        # For each link we store the total sum of bandwidths of the paths crossing each link without middlepoints
        self.compute_link_utilization_reset_sp()

        # We restore the best routing configuration from the DRL agent
        for key, middlepoint in self.sp_middlepoints.items():
            source = int(key.split(':')[0])
            dest = int(key.split(':')[1])
            if middlepoint!=dest:
                # First we remove current routing and then we assign the new middlepoint
                self.decrease_links_utilization_sp(source, dest, source, dest)

                # First we allocate until the middlepoint
                self.allocate_to_destination_sp(source, middlepoint, source, dest)
                # Then we allocate from the middlepoint to the destination
                self.allocate_to_destination_sp(middlepoint, dest, source, dest)        

        # We iterate over all links in an ordered fashion and store the features to edge_state
        self.edgeMaxUti = (0, 0, 0)
        # This list is used to obtain the top K flows from the critical links
        list_link_uti_id = list()
        for i in self.graph:
            for j in self.graph[i]:
                position = self.edgesDict[str(i)+':'+str(j)]
                self.edge_state[position][0] = self.graph[i][j][0]['utilization']
                self.edge_state[position][1] = self.graph[i][j][0]['capacity']
                link_capacity = self.links_bw[i][j]
                # We store the link utilization and the corresponding edge
                list_link_uti_id.append((i, j, self.edge_state[position][0]))

                norm_edge_state_capacity = self.edge_state[position][0]/link_capacity
                if norm_edge_state_capacity>self.edgeMaxUti[2]:
                    self.edgeMaxUti = (i, j, norm_edge_state_capacity)
        
        list_link_uti_id = sorted(list_link_uti_id, key=lambda tup: tup[2], reverse=True)[:self.num_critical_links]
        self._get_top_k_critical_flows(list_link_uti_id)

        # If we want to take the x% bigger demands
        # self.list_eligible_demands = sorted(list_link_uti_id, key=lambda tup: tup[0], reverse=True)
        # self.list_eligible_demands = self.list_eligible_demands[:int(np.ceil(self.numNodes*(self.numNodes-1)*self.percentage_demands))]

        return -self.edgeMaxUti[2]
    
    def allocate_to_destination_sp(self, src, dst, init_source, final_destination): 
        # In this function we allocated the bandwidth by segments. This funcion is used when we want
        # to allocate from a src to a middlepoint and then from middlepoint to a dst using the sp
        bw_allocate = self.TM[init_source][final_destination]
        currentPath = self.shortest_paths[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.graph[firstNode][secondNode][0]['utilization'] += bw_allocate  
            self.graph[firstNode][secondNode][0]['crossing_paths'][str(init_source)+':'+str(final_destination)] = bw_allocate
            self.edge_state[self.edgesDict[str(firstNode)+':'+str(secondNode)]][0] = self.graph[firstNode][secondNode][0]['utilization']
            i = i + 1
            j = j + 1
