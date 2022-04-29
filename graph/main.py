import sys
import os
import subprocess
# distributed processes with multiprocessing lib
# TODO https://zditect.com/code/python/python-multiprocess-process-pool-data-sharing-process-communication-distributed-process.html
import multiprocessing as mp
import networkx as nx
import pandas as pd
import numpy as np
import math
import time
import yaml
import argparse
# libs
import freq_dag

############
# setup related functions
############

def argument_parsing():
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('sw_file', type=str, 
                        help='YAML software definition file.'
                        )
    parser.add_argument('hw_file', type=str, 
                        help='YAML hardware definition file.'
                        )
    # by default, it uses 80% of the cores
    cores = int(math.floor(mp.cpu_count() * 0.8))
    parser.add_argument('-p','--processes', type=int, default=cores,
                        dest='processes', help='Number of working processes.'
                        )
    parser.add_argument('-i','--iter', type=int, default=99999999,
                        dest='iter', help='Max number of iteration executed by a working process.'
                        )
    parser.add_argument('-d','--debug', dest='debug', action='store_true',
                        help='Debug mode.'
                        )
    args = parser.parse_args()

    if not os.path.isfile(args.sw_file):
        print ("ERROR: File %s not found" % (args.sw_file))
        sys.exit(1)
    if not os.path.isfile(args.hw_file):
        print ("ERROR: File %s not found" % (args.hw_file))
        sys.exit(1)
    
    return args

# Loads the software definition
# Returns the list of NetworkX DAGs built from the sw YAML file
def load_sw(filename) -> list():

    # main list with all the DAGs defined in the sw YAML file
    dags = []
    # the YAML file
    sw = None
    with open(filename) as file:
        try:
            sw = yaml.safe_load(file)   
        except yaml.YAMLError as exc:
            print(exc)

    # load the task definition
    print ('TASKs:')
    for idx, i in enumerate(sw['tasks']):
        print ('Task:',idx)
        print (' -',i)
        # 'wcet_ref' is mandatory for a task and must be int >= 0 
        if 'wcet_ref' in i:
            if not isinstance(i['wcet_ref'], int):
                print("ERROR: the 'wcet_ref' attribute expects int data")
                sys.exit(1)
            if i['wcet_ref'] < 0:
                print("ERROR: the 'wcet_ref' attribute must be >= 0")
                sys.exit(1)
        else:
            print("ERROR: the 'wcet_ref' attribute is mandatory for a task")
            sys.exit(1)
        # 'wcet_ref_ns' is mandatory for a task and must be int >= 0 
        if 'wcet_ref_ns' in i:
            if not isinstance(i['wcet_ref_ns'], int):
                print("ERROR: the 'wcet_ref_ns' attribute expects int data")
                sys.exit(1)
            if i['wcet_ref_ns'] < 0:
                print("ERROR: the 'wcet_ref_ns' attribute must be >= 0")
                sys.exit(1)
        else:
            print("ERROR: the 'wcet_ref_ns' attribute is mandatory for a task")
            sys.exit(1)


    # load the DAG definition
    print ('')
    print ('DAGs:')
    for idx, i in enumerate(sw['dags']):
        G = None
        print ('DAG:',idx)
        # check DAG attributes
        # 'activation_period' is mandatory for a DAG and must be int > 0 
        if 'activation_period' in i:
            if not isinstance(i['activation_period'], int):
                print("ERROR: the 'activation_period' attribute expects int data")
                sys.exit(1)
            if i['activation_period'] <= 0:
                print("ERROR: the 'activation_period' attribute must be >= 0")
                sys.exit(1)
        else:
            print("ERROR: the 'activation_period' attribute is mandatory for a DAG")
            sys.exit(1)
        # 'deadline' is mandatory for a DAG and must be int > 0 
        if 'deadline' in i:
            if not isinstance(i['deadline'], int):
                print("ERROR: the 'deadline' attribute expects int data")
                sys.exit(1)
            if i['deadline'] <= 0:
                print("ERROR: the 'deadline' attribute must be >= 0")
                sys.exit(1)
        else:
            print("ERROR: the 'deadline' attribute is mandatory for a DAG")
            sys.exit(1)
        # 'deadline' must be <= 'activation_period'
        if i['deadline'] > i['activation_period']:
            print("ERROR: the 'deadline' attribute must be <= 'activation_period'")
            sys.exit(1)
        # DAG edges
        print (' - edges:',len(i['edge_list']))
        edge_list = []
        for e in i['edge_list']:
            if not isinstance(e[0], int) or not isinstance(e[1], int):
                print("ERROR: the graph edges must be int")
                sys.exit(1)
            # the edges must be between [0,#tasks-1]
            if e[0] < 0 or e[1] < 0 or e[0] >= len(sw['tasks']) or e[1] >= len(sw['tasks']):
                print("ERROR: the graph edges must be between >= 0 and <= #tasks-1")
                sys.exit(1)
            print("   -", tuple(e))
            edge_list.append(tuple(e))
        sources = [x[0] for x in edge_list]
        targets = [x[1] for x in edge_list]
        weights = [0 for x in range(len(edge_list))] # defined in runtime
        linkData = pd.DataFrame({'source' : sources,
                        'target' : targets,
                        'weight' :weights})

        # DAG nodes
        # use the set to get unique node ids
        node_names = {e for l in edge_list for e in l}
        # then get ride of the set and transform it into a array
        node_names = list(node_names)
        print (' - node names:',node_names)
        wcet = [sw['tasks'][t]['wcet_ref'] for t in range(len(sw['tasks'])) if t in node_names]
        wcet_ns = [sw['tasks'][t]['wcet_ref_ns'] for t in range(len(sw['tasks'])) if t in node_names]
        zeros = [0]*len(node_names)
        nodeData = pd.DataFrame({'name' : node_names,
                        'wcet_ref' : wcet,     # wcet at the reference freq
                        'wcet_ref_ns' : wcet_ns, # non scalable part of the wcet
                        'rel_deadline': zeros, # defined in runtime
                        'wcet': zeros # defined in runtime
                        })
        G = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())
        nx.set_node_attributes(G, nodeData.set_index('name').to_dict('index'))
        # transform the graph into adjacency_matrix for better printing
        # sparse_adj_mat = nx.adjacency_matrix(G)
        # array_adj_mat = sparse_adj_mat.toarray('C')
        # print (array_adj_mat)

        # other DAG attributes
        print (' - activation_period:',i['activation_period'])
        print (' - deadline:',i['deadline'])
        # print (' - ref_freq:',i['ref_freq'])
        G.graph['activation_period'] = i['activation_period']
        G.graph['deadline'] = i['deadline']
        # the reference freq is the freq used to characterize this application
        # G.graph['ref_freq'] = i['ref_freq']
        # when it's deadling with multiple DAGs, this will find the first and last nodes of a DAG
        last_task = [t for t in G.nodes if len(list(G.successors(t))) == 0]
        first_task = [t for t in G.nodes if len(list(G.predecessors(t))) == 0]
        if len(first_task) != 1:
            print('ERROR: invalid DAG with multiple starting nodes')
            sys.exit(1)
        if len(last_task) != 1:
            print('ERROR: invalid DAG with multiple ending nodes')
            sys.exit(1)
        # the dummy tasks representing the start and finish nodes
        G.graph['first_task'] =  first_task[0]
        G.graph['last_task'] = last_task[0]
        # the 1st and last tasks must have, respectively, the lowest and the highest node index
        if G.graph['first_task'] != min(G.nodes):
            print('ERROR: the 1st task does not have the lowest index of the DAG')
            sys.exit(1)
        if G.graph['last_task'] != max(G.nodes):
            print('ERROR: the last task does not have the highest index of the DAG')
            sys.exit(1)
        # this is the set of tasks, excluding the dummy tasks
        G.graph['actual_tasks'] = set([n for n in G.nodes]) - set([G.graph['first_task']]) - set([G.graph['last_task']])
        # save it into the list of DAGs
        dags.append(G)

    # this limitation is related to the way the placements are encoded using an 'int'
    # TODO test this limit using a 'long int' to support up to 64 tasks
    n_tasks = sum([len(G.graph['actual_tasks']) for G in dags])
    if n_tasks > 32:
        print ('ERROR: Currently up to 32 tasks are supported')
        sys.exit(1)

    for G in dags:
        print ("dag properties:")
        print (G.graph)

        print ("node properties:")
        for k,v in G.nodes(data=True):
            print(k,v)

        # print ("edge properties:")
        # for n1, n2, data in G.edges(data=True):
        #     print(n1, n2, data)

    return dags

# Loads the hardware platform definition.
# Returns the YAML file representing the hardware platform spec.
def load_hw(filename):
    # YAML with the hardware definition file
    islands = None
    with open(filename) as file:
        try:
            islands = yaml.safe_load(file)   
        except yaml.YAMLError as exc:
            print(exc)

    # load power-related data from the CSV into the islands list of dict
    print ('')
    print ('ISLANDS:')
    for i in islands['hw']:
        #######################
        # check the power file
        #######################
        if 'power_file' not in i:
            print("ERROR: the 'power_file' attribute is mandatory for an island")
            sys.exit(1)
        if not os.path.isfile(i['power_file']):
            print ('ERROR: file', i['power_file'], 'not found')
            sys.exit(1)
        data=pd.read_csv(i['power_file'])
        if  len(data.keys()) != 3 or data.keys()[0] != 'freq' or data.keys()[1] != 'busy_power_avg' or data.keys()[2] != 'idle_power_avg':
            print ('ERROR: file', i['power_file'], 'has an invalid syntax')
            sys.exit(1)
        #######################
        # check input data types
        #######################
        # 'capacity' is mandatory and must be float >0.0 and <= 1.0 
        if 'capacity' in i:
            if not isinstance(i['capacity'], float):
                print("ERROR: the 'capacity' attribute expects float data")
                sys.exit(1)
            if i['capacity'] <= 0.0 or i['capacity'] > 1.0:
                print("ERROR: the 'capacity' attribute must be between (0.0, 1.0]")
                sys.exit(1)
        else:
            print("ERROR: the 'capacity' attribute is mandatory for an island")
            sys.exit(1)
        # 'n_pus' is mandatory and must be int > 0
        if 'n_pus' in i:
            if not isinstance(i['n_pus'], int):
                print("ERROR: the 'n_pus' attribute expects int data")
                sys.exit(1)
            if i['n_pus'] <= 0:
                print("ERROR: the 'n_pus' attribute must be > 0")
                sys.exit(1)
        else:
            print("ERROR: the 'n_pus' attribute is mandatory for an island")
            sys.exit(1)
        # it is IMPORTANT that the power data is sorted in ascending frequency
        data.sort_values(by=['freq'], inplace=True)
        i['busy_power'] = list(data.busy_power_avg)
        i['idle_power'] = list(data.idle_power_avg)
        i['freqs'] = list(data.freq)
        # the reference frequency must exist only in the higher capacity island
        if 'ref_freq' in i:
            if not isinstance(i['ref_freq'], int):
                print("ERROR: the 'ref_freq' attribute expects integer data")
                sys.exit(1)
            if i['capacity'] != 1.0:
                print("ERROR: the 'ref_freq' attribute is expected only int the high capacity island")
                sys.exit(1)
            if i['ref_freq'] not in i['freqs']:
                print("ERROR: the 'ref_freq' attribute must match one of the existing frequencies")
                sys.exit(1)
        # i['placement'] = [] # the tasks assigned to this island
        # i['pu_utilization'] = [0.0]*i['n_pus'] # the utilization on each PU
        # i['pu_placement'] = [[] for aux in range(i['n_pus'])] # the tasks assigned to each PU
        del(i['power_file'])

    for idx, i in enumerate(islands['hw']):
        print ('island:', idx)
        for key, value in i.items():
            print(" -", key, ":", value)

    # sort the islands by capacity. also drop the 'hw' level from it
    islands = sorted(islands['hw'], key = lambda ele: ele['capacity'])

    return islands

# The 'unrelated_task_set' is used to account the DAG precedence constraint 
# into PU utilization calculation. 
# It returns a list of lists where the first and last nodes are excluded
# TODO: implements Tommaso's code
def create_unrelated_task_set(dags) -> list():
    # used to account the DAG precedence constraint into PU utilization calculation
    # the first and last nodes are excluded
    # unrelated = [
    #     [ 6, 7, 8 ], 
    #     [ 3, 6 ], 
    #     [ 5, 6, 8 ], 
    #     [ 1, 5 ], 
    #     [ 4, 5 ], 
    #     [ 1, 2, 3 ], 
    #     [ 3, 4 ]
    # ]
    # use this when running the example w 2 dags
    unrelated = [
        [ 6, 7, 8, 11 ], 
        [ 3, 6, 11 ], 
        [ 5, 6, 8, 11 ], 
        [ 1, 5, 11 ], 
        [ 4, 5, 11 ], 
        [ 1, 2, 3, 11 ], 
        [ 3, 4, 11 ]
    ]
    # TODO sort 'unrelated' by deacreasing wcet assuming the highest frequency of the high capacity island.
    # This will increase the probability of find failing set in the first iterations in function 'check_utilization'

    print ("unrelated sets:")
    for u in unrelated:
        print (u)

    return unrelated

# subdivide the entire search space into work packages that will
# be executed by the pool of processes
def setup_workload(dags, n_procs, max_placements) -> list():
    n_tasks = sum([len(G.graph['actual_tasks']) for G in dags])
    # the size of the entire search space of 'n_tasks' placements onto 'n_islands'
    total_task_placements = n_islands**n_tasks
    # it represents how many work packages i can divide the entire search space (total_task_placements)
    # of placements considering a parallelism of 'n_threads'
    n = 1
    while (n_procs > n_islands**n):
        n=n+1
    search_space_subdivisions = n_islands**n
    # the size of each work package, i.e., the number of placements to be checked by work package
    placements_per_workload = int (math.ceil(float(total_task_placements) / float(search_space_subdivisions)))
    print ('total_task_placements:',total_task_placements)
    print ('search_space_subdivisions:', search_space_subdivisions)
    print ('placements_per_workload:', placements_per_workload)

    # Generate the data structure sent to the pool of processes, i.e., the initial placement and 
    # how many placements each work package must check, and the list of DAGs. 
    # The way it's calculated, all work package should have the same size, regardeless 
    # n_threads, n_islands, n_tasks
    placement_cnt = 0
    placement_setup_list = []
    while (placement_cnt < total_task_placements):
        # initial placement id and the number of placements to be generated from this initial one
        placement_setup_list.append((placement_cnt, min(placements_per_workload,max_placements), dags))
        placement_cnt += placements_per_workload
    print ('work packages:', search_space_subdivisions,'size:', min(placements_per_workload,max_placements))

    return placement_setup_list

############
# optimization related functions
############

# return power consumed by the island i
# TODO try a matrix-based version of this function
def island_power(i,dags, placement, freqs_per_island_idx) -> float:
    # get the index of the tasks deployed in island i
    # assumes the tasks where already placed on the islands
    deployed_tasks = list(placement[i])
    
    # get the assigned freq and scales it down linearly with the the island max frequency
    busy_power = islands[i]['busy_power'][freqs_per_island_idx[i]]
    idle_power = islands[i]['idle_power'][freqs_per_island_idx[i]]

    utilization = 0.0
    for t in deployed_tasks:
        activation_period  = 0
        wcet = 0
        # find to which DAG this task belongs to
        for G in dags:
            if t in G.graph['actual_tasks']:
                activation_period = G.graph['activation_period']
                # assumes that wcet was calculated previously
                wcet = G.nodes[t]['wcet']
                break
        if activation_period != 0:
            utilization = utilization + (float(wcet)/float(activation_period))
        else:
            print ("ERROR: not expecting to reach this point in 'island_power' function")
            sys.exit(1)
    # About this part (islands[i]['n_pus'] * idle_power)
    # The rational is that, even when the utilization is 0, we have to account for the idle power of each pu of this island
    island_total_power = (islands[i]['n_pus'] * idle_power) + (busy_power-idle_power) * float(utilization)

    return island_total_power


# sum up the power of each island based on current task placement and island frequency
def define_power(dags, placement, freqs_per_island_idx) ->  float:
    total_power = 0.0
    for i in range(n_islands):
        total_power = total_power +  island_power(i,dags, placement, freqs_per_island_idx)
    return total_power

# Define the wcet of each task based on which island the task is placed.
def define_wcet(dags, placement, freqs_per_island_idx) -> None:
    # wcet for all tasks assigned to 0
    for G in dags:
        for t in G.nodes:
            G.nodes[t]["wcet"] = 0
    # reference frequency is an attribute of the hardware 
    # used to extract the performance parameters of the sw YAML file
    f_ref = islands[-1]['ref_freq']
    # calculate the wcet for each task
    for idx, i in enumerate(islands):
        # get the frequency assigned to the island i
        f = i['freqs'][freqs_per_island_idx[idx]]
        capacity = i['capacity']
        # for t in i['placement']:
        for t in placement[idx]:
            # find to which DAG this task belongs to
            G = None
            for dag in dags:
                if t in dag.graph['actual_tasks']:
                    G = dag
                    break
            wcet_ref_ns = G.nodes[t]['wcet_ref_ns']
            wcet_ref = G.nodes[t]['wcet_ref']
            wcet = wcet_ref_ns + float((wcet_ref-wcet_ref_ns)/f * f_ref)/float(capacity)
            G.nodes[t]["wcet"] = int(math.ceil(wcet))
    # cannot have a task not placed in an island
    for G in dags:
        for t in G.graph['actual_tasks']:
            if G.nodes[t]["wcet"] == 0:
                print ('ERROR: wcet for task', t, 'not defined')
                sys.exit(1)

        # the 1st and last nodes have no computation
        G.nodes[G.graph['first_task']]["wcet"] = 0
        G.nodes[G.graph['last_task']]["wcet"] = 0

# assign path wcet to each node
# TODO the underlaying algorithm is graph-related search
# which is not scalable code with compexity O(n!)
# return the list of path wcet for each node. return an empty list if 
# the dag deadline is not feasible
def define_path_wcet(H) -> list():
    # main steps:   
    # 1) convert node weight into edge weight to find longest paths
    # 2) get all paths to each end node
    # 3) for each node, assign its max path wcet

    ####################
    # 1) convert node weight into edge weight to find longest paths
    ####################
    H.nodes[H.graph['first_task']]["wcet"] = 0
    H.nodes[H.graph['last_task']]["wcet"] = 0
    dag_deadline = H.graph['deadline']

    # get max node weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])
    # start_time = time.time()
    max_task_wcet = max([H.nodes[n]["wcet"] for n in H.nodes])
    # elapsed_time = time.time() - start_time

    # if a single task is longer than the dag deadline, then this is not a solution
    if (max_task_wcet > dag_deadline):
        if debug:
            print ('WARNING: a single task wcet is longer than then DAG deadline', dag_deadline)
            for t in H.graph['actual_tasks']:
                print (t, H.nodes[t]["wcet"])
        # temp_tuple = (terminate_counters[0][0]+1,terminate_counters[0][1]+elapsed_time)
        # terminate_counters[0] = temp_tuple
        return []

    # assign the edge weight as the sum of the node weights
    for u, v, data in H.edges(data=True):
        # invert the edge weight since we are looking for the longest path
        data['weight'] = max_weight - (H.nodes[u]["wcet"] + H.nodes[v]["wcet"])

    for n in H.nodes:
        H.nodes[n]["rel_deadline"] = 0

    # 
    # This is the problem with shortest path for this problem:
    # 
    # PARTIAL PATH:
    # (84, [0, 3, 8]) [0, 52, 32]
    # 0 3 {'weight': 32}
    # 3 8 {'weight': 0}
    # 8 9 {'weight': 52}
    # (116, [0, 3, 5, 7]) [0, 52, 32, 32]
    # 0 3 {'weight': 32}
    # 3 5 {'weight': 0}
    # 5 7 {'weight': 20}  <=== problem
    # 7 9 {'weight': 52}
    # 
    # This will make the algorithm not find the critical path since the edge 
    # weights are not correctly assigned

    #start_time = time.time()
    critical_path = nx.shortest_path(H,H.graph['first_task'],H.graph['last_task'],weight='weight')
    wcet_critical_path = sum([H.nodes[n1]["wcet"] for n1 in critical_path])
    #elapsed_time = time.time() - start_time
    #print ('CRITICAL PATH:')
    #print (wcet_critical_path,critical_path)
    if wcet_critical_path > dag_deadline:
        if debug:
            print ('WARNING: critical path', critical_path,'has lenght', wcet_critical_path, 'which is longer than then DAG deadline', dag_deadline)
        # temp_tuple = (terminate_counters[1][0]+1,terminate_counters[1][1]+elapsed_time)
        # terminate_counters[1] = temp_tuple            
        return []

    ####################
    # 2) get all paths to each end node
    ####################
    # get the last edges and nodes
    last_edges = H.in_edges(H.graph['last_task'])
    last_nodes = [e[0] for e in last_edges]
    # get the paths to the last nodes
    # TODO: bad scalability !!! the 'all_simple_paths' function is O(n!) in the complete graph of order n.
    paths_from_last_nodes = []
    all_paths_list = []
    for n in last_nodes:
        path_list = []
        # get all the paths to last node n
        paths = nx.all_simple_paths(H, H.graph['first_task'], n)
        # make it a list of paths
        path_list.extend(paths)
        paths_from_last_nodes.append(path_list)
        all_paths_list = all_paths_list + path_list

    ####################
    # 3) for each node, assign its max path wcet
    ####################
    H.nodes[H.graph['first_task']]["wcet"] = 0
    H.nodes[H.graph['last_task']]["wcet"] = 0
    # create a list with node wcet
    path_wcet_list = [H.nodes[n]["wcet"] for n in H.nodes]
    # a tuple to save the longest of all paths. format (path wcet, path)
    critical_path2 = (0,[])
    for n in H.graph['actual_tasks']:
        #start_time = time.time()
        # get all the paths where node n is found
        paths_of_node_n = [p for p in all_paths_list if n in p]
        # get the wcet for each path
        path_wcet_sum = []
        for p in paths_of_node_n:
            # make a tuple w the sum of the path and the path
            path_wcet_sum.append((sum([H.nodes[n1]["wcet"] for n1 in p]), p))
        # get the path with the longest wcet
        max_partial_path = max(path_wcet_sum,key=lambda item:item[0])
        #elapsed_time = time.time() - start_time
        # if a path was found longer than the DAG deadline, this cannot be a feasible solution
        if max_partial_path[0] > dag_deadline:
            if debug:
                print ('WARNING: critical path', max_partial_path[1], 'takes', max_partial_path[0],', longer than DAG deadline', dag_deadline)
            # temp_tuple = (terminate_counters[2][0]+1,terminate_counters[2][1]+elapsed_time)
            # terminate_counters[2] = temp_tuple
            return []
        # save the longest of all paths
        if max_partial_path[0] > critical_path2[0]:
            critical_path2 = max_partial_path
        # mark each node of the selected path as 'path_wcet_list'
        for p in max_partial_path[1]:
            # replace node wcet by path wcet
            # so, each item in this list has its longest path wcet
            # 'p-H.graph['first_task']' is required when working with multi dags
            # since, except for the 1st dag, all the other dags wont have a 1st node of index 0
            offset_idx = p-H.graph['first_task']
            path_wcet_list[offset_idx] = max(max_partial_path[0],path_wcet_list[offset_idx])
    # if the critical path is longer than the DAG deadline, this cannot be a solution
    if critical_path2[0] > dag_deadline:
        # not expecting to reach this point unless there is a bug above
        print ('ERROR: critical path', critical_path2[1], 'takes', critical_path2[0],', longer than DAG deadline', dag_deadline)
        return []

    return path_wcet_list

# Assign relative deadline to each node
# TODO it's also very inneficient because this is calculated for every frequency for every placement.
# the critical path and the relative deadlines could be pre-processed only once at startup
# TODO replace it by Tommaso's function
# freqs_per_island_idx used only for debugging purposes
def define_rel_deadlines(dags,freqs_per_island_idx) -> bool:    
    # main steps:   
    # 1) assign path wcet to each node - compexity O(n!)
    # 2) assign deadline to all nodes proportionally to its wcet and path wcet
    # 3) (DISABLED STEP) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    # 4) transfer the relative deadline back to the original DAG

    for H in dags:
        # some basic definitions
        dag_deadline = H.graph['deadline']

        ############################
        # 1) assign path wcet to each node
        ############################
        path_wcet_list = define_path_wcet(H)
        if not path_wcet_list:
            if debug:
                print ('relative deadline failed with freqs',freqs_per_island_idx)
            # an empty list, which means dag deadline is not feasible
            return False
        #print (path_wcet_list)
        ############################
        # 2) assign deadline to all nodes proportionally to its wcet and path wcet
        ############################
        for n in H.graph['actual_tasks']:
            # 'n-H.graph['first_task']' is required when working with multi dags
            # since, except for the 1st dag, all the other dags wont have a 1st node of index 0
            offset_idx = n-H.graph['first_task']
            wcet_ratio = float(H.nodes[n]["wcet"])/float(path_wcet_list[offset_idx])
            # assign rel_deadline proportional to its wcet
            H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    

        # although this idea of increasing the deadline is valid, 
        # the benefit is low compared to the computational cost of computing 
        # all paths. On the other hand, this trick is able to reduce the utilization 
        # a little bit, which might increase the number of 'potential solutions'
        # Thus, for now this is disabled
        if False:
            ############################
            # 3) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
            ############################
            # So far, it is not garanteed that the nodes have theirs respective maximal relative deadline. This last step does that 
            # by trying to increase the relative deadline of the last nodes.
            for paths in paths_from_last_nodes:
                # the last node index
                node = paths[0][-1]
                rel_deadline_sum = []
                # sum the deadlines for all paths leading to node
                for p in paths:
                    # save the sum the deadlines for path p
                    rel_deadline_sum.append(sum([H.nodes[n1]["rel_deadline"] for n1 in p]))
                # get the path with the longest sum of rel_deadline
                max_rel_deadline_sum = max(rel_deadline_sum)
                if (max_rel_deadline_sum > dag_deadline):
                    # not expecting to reach this point unless there is a bug above
                    print ('ERROR: not expecting to have longer deadlines at this point. Path sum is',max_rel_deadline_sum)
                    sys.exit(1)
                # assign any reamaning slack to its last node
                H.nodes[node]["rel_deadline"] = H.nodes[node]["rel_deadline"] + (dag_deadline - max_rel_deadline_sum)

        # the relative deadline of a task cannot be lower than its wcet
        for n in H.graph['actual_tasks']:
            if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
                if debug:
                    print ('WARNING: task', n, 'has wcet', H.nodes[n]["wcet"], 'and relative deadline', H.nodes[n]["rel_deadline"])
                    print ('relative deadline failed with freqs',freqs_per_island_idx)
                # temp_tuple = (terminate_counters[3][0]+1,0.0)
                # terminate_counters[3] = temp_tuple
                return False

    return True

# Generates the datafile required to run the function 'optimal_placement'
# The generated model is such that all nodes are included into the model 
# instead of only the ones assigned to the island. This way, it's easier 
# to check the precedence constraints. However, the utilization is zero
# when the task is not mapped to the target island.
# TODO update it to work with multiple DAGs
def create_minizinc_datafile(dags, i, placement, freq_seq, filename):

    def unrelated_line(f, line_start, n, unrel_row):
        f.write(line_start)
        for i in range(n):
            if i < len(unrel_row):
                f.write(str(unrel_row[i]+1))
            else:
                f.write("1")
            if i < n-1:
                f.write(",")
        f.write("\n")

    #generate the filename and open the file
    f = None
    try:
        f = open(filename,"w")
    except IOError:
        print ("ERROR: could not generate file", filename)
        sys.exit(1)
    
    f.write("%% The problem instance:\n")
    f.write("%%   island = %d\n" % (i))
    f.write("%%   placement = %s\n" % (str(placement)))
    f.write("%%   freq_seq = %s\n\n" % (str(freq_seq)))
    f.write("N_CORES = %d;\n" % (islands[i]['n_pus']))
    f.write("N_DAGS = %d;\n" % ( len(dags) ))
    # All nodes are included into the model instead of only the ones assigned to the island.
    # This way, it's easier to check the precedence constraints
    f.write("% these two constants represent the sum of all nodes and edges of all DAGs\n")
    f.write("N_NODES = %d;\n" % ( sum([len(G.nodes) for G in dags]) ))
    f.write("N_EDGES = %d;\n\n" % ( sum([len(G.edges) for G in dags]) ))

    f.write("N_UNREL_ROWS = %d;\n" % (len(unrelated)))
    max_unrel_size = max([len(i) for i in unrelated])
    f.write("N_UNREL_COLS = %d;\n\n" % (max_unrel_size))

    f.write("% the deadline of each DAG\n")
    f.write("Ddag = %s;\n\n" % ( [G.graph['deadline'] for G in dags] ))

    f.write("% required to fix the index of the matrices shared among the DAGs\n")
    f.write("first_node = %s;\n" % ( [G.graph['first_task']+1 for G in dags] ))
    f.write("last_node = %s;\n\n" % ( [G.graph['last_task']+1 for G in dags] ))

    # gather the utilization required from each task considering all DAGs
    task_utilization = []
    for G in dags:
        first_task = G.graph['first_task']
        for n in range(len(G.nodes)):
            util = 0.0
            # if the node n is not in the task set to be placed in this island,
            # then utilization is 0, i.e., it wont be taken into account when 
            # cheking the pu utilization
            if first_task+n in placement[i]:
                if G.nodes[first_task+n]['rel_deadline'] != 0:
                    util = float(G.nodes[first_task+n]['wcet'])/float(G.nodes[first_task+n]['rel_deadline'])
            task_utilization.append(round(util,4))

    f.write("% relative deadline to each task\n")
    rel_dealine = []
    for G in dags:
        first_task = G.graph['first_task']
        for n in range(len(G.nodes)):
            rel_dealine.append (G.nodes[first_task+n]['rel_deadline'])
    f.write("D = %s;\n\n" % ( rel_dealine ))

    f.write("% utilization required by task: T[i]/D[i]\n")
    f.write("U = %s;\n\n" % (task_utilization))

    f.write("% non related nodes, i.e., nodes that could be executed concurrently\n")
    f.write("unrelated_node = \n")
    # the 1st line
    unrelated_line(f,"   [| ",max_unrel_size,unrelated[0])
    # the middle lines
    for row in unrelated[1:]:
        unrelated_line(f,"    | ",max_unrel_size,row)
    # the last line
    f.write("    |];\n\n")

    f.write("% from: the leaving node for each edge\n")
    f.write("% to: the entering node for each edge\n")
    f.write("% list of edges indicating which nodes are connected\n")
    f.write("E =  \n")
    idx = 0
    for G in dags:
        for n1, n2 in G.edges():
            if idx == 0:
                f.write("   [")
            else:
                f.write("    ")
            f.write("| "+str(n1+1)+","+str(n2+1)+"\n")
            idx = idx = 1
    # the last line
    f.write("    |];\n")

    f.close()

# Optimal solution to the place a task set onto PUs.
# It returns true if the task placement is feasible.
# TODO capture the placement output of minizinc
# TODO update it to work with multiple DAGs
def optimal_placement(dags, i, placement, freq_seq) -> bool:
    global minizinc_cnt
    minizinc_cnt = minizinc_cnt +1

    # data_filename  = str(placement)+"-"+str(freq_seq)+".dnz"
    # data_filename  = data_filename.replace(" ", "")
    data_filename = 'data_gen.dzn'

    create_minizinc_datafile(dags, i, placement, freq_seq, data_filename)

    # print ("Running minizinc:", data_filename)
    proc_minizinc = None
    try:
        proc_minizinc = subprocess.Popen(["minizinc", 
            "model.mzn", 
            data_filename],stdout=subprocess.PIPE,
            stderr=subprocess.PIPE )
    except subprocess.CalledProcessError as e:
        print(e.output)
        print ("ERROR: could not call minizinc. Perhaps it is not installed ?!?!")
        print ("Error running 'minizinc model.mzn %s'" % data_filename)
        sys.exit(1)

    # CONVERT THE MINIZINC OUTPUT INTO JSON 
    # the minizinc output is a list of bytes
    minizinc_stdout_bytes = proc_minizinc.stdout.readlines()
    # this will convert a list of bytes into a single string
    minizinc_stdout_str=b''.join(minizinc_stdout_bytes).decode('utf-8')
    if "UNSATISFIABLE" in minizinc_stdout_str:
        # print ('MINIZINC: False')
        # print (placement, freq_seq)
        return False
    else:
        # print ('MINIZINC: True')
        # print (minizinc_stdout_str)
        return True

# It initially performs a worst_fit greedy approach to place the tasks of an island onto PUs.
# If the heuristic says that it is not possible to place the tasks, then it runs an exact optimal
# solver to confirm it. Hopefully, the heuristic will be sufficient most of the times.
# Returns false if any PU on any island exeeds the utilization threshold.
def check_utilization(dags, placement, freqs_per_island_idx) -> bool:
    global unrelated
    # for each island, run the placement heuristic and, if required, the exact solution
    # The 'unrelated' set represents the list of tasks that could be run concurently in a PU
    for u in unrelated:
        u_set = set(u)
        for idx, i in enumerate(islands):
            p_set = set(placement[idx])
            # the tasks in this island that are also in the unrelated set
            u_set_in_island = u_set.intersection(p_set)
            # no need to perform utilization analysis if there is only one task in this island.
            # the way it is calculated, a single task will always fit in an island
            if len(u_set_in_island)<=1:
                continue
            # to keep track of worst_fit heuristic with good data locality
            utilization_per_pu = [0.0]*i['n_pus']
            task_placement = [[] for aux in range(i['n_pus'])]
            # have to calculate the task utilization upfront in order to sort the tasks by utilization,
            # increasing the efficienty of the worst-fit heurisitic, reducing the need to run the optimal_placement function
            task_utilization = []
            # fill the list of tuple (task_utilization, task id)
            for t in u_set_in_island:
                # find to which DAG this task belongs to
                G = None
                for dag in dags:
                    if t in dag.graph['actual_tasks']:
                        G = dag
                        break
                task_utilization.append((float(G.nodes[t]['wcet']) / float(G.nodes[t]['rel_deadline']), t))
            # the longest tasks are placed first
            task_utilization.sort(key=lambda y: y[0], reverse=True)
            for t in task_utilization:
                # get the PUs with minimal utilization
                pu = utilization_per_pu.index(min(utilization_per_pu))
                # check if it is possible to assign this task to the pu, i.e., if the pu utilization is < 1.0
                if utilization_per_pu[pu] + t[0] > 1.0:
                    if debug:
                        print ('WARNING: utilization constraint failed',freqs_per_island_idx, task_utilization)
                    # run minizinc to confirm whether it is indeed impossible to have this set of tasks placed on these PUs
                    return optimal_placement(dags, idx, placement, freqs_per_island_idx)
                    # return False
                else:
                    utilization_per_pu[pu] = utilization_per_pu[pu] + t[0]
                    task_placement[pu].append(t[1])

    # TODO is it required to copy the solution back to the islands list ?
    # the solution is copied back to the islands data structure only if the task placement is feasible for all islands
    # perhaps the best is to find the best solutions, and then calculate it again
    # for idx, i in enumerate(islands):
    #     i['pu_utilization'] = list(temp_solution[idx][0])
    #     i['pu_placement'] = list(temp_solution[idx][1])
    return True

# 2nd implementation of check_utilization() where the task utilization
# was calculated before hand
# TODO update it to multiple DAGs
def check_utilization_mat(task_utilization_array) -> bool:
    # for each island, run the placement heuristic and, if required, the exact solution
    temp_solution = []
    # first_task = G.graph['first_task']
    # last_task = G.graph['last_task']
    for idx, i in enumerate(islands):
        # to keep track of worst_fit heuristic with good data locality
        utilization_per_pu = [0.0]*i['n_pus']
        task_placement = [[] for aux in range(i['n_pus'])]
        # remove the initial and last nodes of the DAG
        #task_set = [t for t in i['placement'] if t != first_task and t != last_task]
        # get the task placement on islands by getting the index of task_utilization_array
        # where the value is > 0.0
        task_set = np.argwhere(task_utilization_array[idx]>0.0).transpose()[0].tolist()
        # TODO would it be better to sort the array in deacreasing task utilization ?
        for t in task_set:
            # get the PUs with minimal utilization
            pu = utilization_per_pu.index(min(utilization_per_pu))
            # get the utilization for the current task t
            # pu_utilization = (float(G.nodes[t]['wcet']) / float(G.nodes[t]['rel_deadline']))
            #print (t, pu, pu_utilization)
            pu_utilization = task_utilization_array[idx,t]
            # check if it is possible to assign this task to the pu, i.e., if the pu utilization is < 1.0
            if pu_utilization == 0.0:
                print ('ERROR: not expecting to reach this point')
                sys.exit(1)
            if utilization_per_pu[pu] + pu_utilization > 1.0:
                # run minizinc to confirm whether it is indeed impossible to have this set of tasks placed on these PUs
                # return optimal_placement(idx,freqs_per_island_idx)
                return False
                # task_placement = []
                # if len(task_placement) == 0:
                #     return False
                # break
            else:
                utilization_per_pu[pu] = utilization_per_pu[pu] + pu_utilization
                task_placement[pu].append(t)
        temp_solution.append((utilization_per_pu,task_placement))
    # the solution is copied back to the islands data structure only if the task placement is feasible for all islands
    for idx, i in enumerate(islands):
        i['pu_utilization'] = list(temp_solution[idx][0])
        i['pu_placement'] = list(temp_solution[idx][1])
    
    return True

# This is a matrix-based (i.e. similar to matlab code) for cheking
# whether the task placement is viable, i.e., the critical path 
# does not exceed the DAG deadline and PU utilization is below 1.0
# TODO update it to work w multiple DAGs
def check_placement_mat(placement_array) -> bool:
    # This procedure is divided into the following parts
    # 1) WCET definition for each task
    # 2) assign path wcet to each node and check the critical path against the dag deadline
    # 3) calculate relative deadline for each task
    # 4) check PU utilization constraint

    ###########################
    # 1) WCET definition for each task
    ###########################
    # This block of code is equivalent to the following procedural code:
    # for idx, i in enumerate(islands):
    #     # get the frequency assigned to the island i
    #     f = i['freqs'][freqs_per_island_idx[idx]]
    #     capacity = i['capacity']
    #     for t in i['placement']:
    #         wcet_ref_ns = G.nodes[t]['wcet_ref_ns']
    #         wcet_ref = G.nodes[t]['wcet_ref']
    #         wcet = wcet_ref_ns + (capacity * (wcet_ref-wcet_ref_ns)/f * f_ref)
    #         G.nodes[t]["wcet"] = int(math.ceil(wcet))

    n_nodes = len(G.nodes)
    if debug:
        print('placement')
        print (placement_array)

    ################
    ##### these are the island-related matrices, defined when the frequency changes
    ################
    # 1 x n_islands matrix with the frequencies assigned to each island, in this case, the highest frequency for each island
    max_freq_array = np.asarray([[i['freqs'][-1] for i in islands]])
    ################
    ##### these are the island-related matrices, defined once at the initialization 
    ################
    # the scalar ref_freq is expanded into 1 x n_islands matrix
    f_ref_array    = np.asarray([[G.graph['ref_freq'] for i in range(n_islands)]])
    # 1 x n_islands matrix with the capacity of each island
    capacity_array = np.asarray([[i['capacity'] for i in islands]])
    # this is the part of the wcet calculation that does not depend on the task
    f_ratio_array  = capacity_array * f_ref_array / max_freq_array
    # repeat the last column by the number of nodes so it is possible to perform operations with the task-related matrices
    # the result is a n_islands x n_nodes matrix
    expanded_ratio = np.asarray([[f_ratio_array[0,i]]*n_nodes for i in range(n_islands)])

    ################
    ##### these are the task-related matrices, defined every new candidate task placement
    ################
    # n_islands x n_nodes matrix. Since these 2 matrices are measured using the ref_freq, not the assigned freq,
    # the result is that the matrix has reapted rows
    wcet_ref_ns_array = np.asarray([[G.nodes[t]['wcet_ref_ns'] for t in range(len(G.nodes))] for i in range(n_islands)])
    wcet_ref_array    = np.asarray([[G.nodes[t]['wcet_ref']    for t in range(len(G.nodes))] for i in range(n_islands)])
    wcet_delta_array  = wcet_ref_array-wcet_ref_ns_array
    # take into account the placement matrix such that the non-zero values represent the correct island placement
    # n_islands x n_nodes matrices
    wcet_delta_placement_array  = placement_array * wcet_delta_array
    wcet_ref_ns_placement_array = placement_array * wcet_ref_ns_array
    # the final wcet for all tasks in all islands
    wcet_array = wcet_ref_ns_placement_array + wcet_delta_placement_array * expanded_ratio
    # rouding up to have integer wcet
    wcet_array = np.ceil(wcet_array).astype(int)
    # copy the wcet to the nodes
    for idx, row in enumerate(wcet_array):
        task_set = np.argwhere(row>0.0).transpose()[0].tolist()
        for t in task_set: 
            G.nodes[t]["wcet"] = wcet_array[idx,t]
    if debug:
        print ('wcet')
        print (wcet_array)

    ############################
    # 2) assign path wcet to each node and check the critical path against the dag deadline
    ############################
    # deep copy the DAG
    H = G.copy()
    path_wcet_list = define_path_wcet(H)
    if not path_wcet_list:
        # an empty list, which means dag deadline is not feasible
        return False
    #print (path_wcet_list)
    # get the longest path related to each node
    path_wcet_array = np.asarray(path_wcet_list)
    path_wcet_array = path_wcet_array.astype(float)

    ###########################
    # 3) calculate relative deadline for each task
    ###########################
    # The following code is equivalent to this procedural code one:
    # remove the initial and last nodes of the DAG
    # task_set = [t for t in H.nodes if t != H.graph['first_task'] and t != H.graph['last_task']]
    # for n in task_set:
    #     wcet_ratio = float(H.nodes[n]["wcet"])/float(path_wcet_list[n])
    #     H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    
    # for n in task_set:
    #     if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
    #         return False
    dag_deadline = G.graph['deadline']
    wcet_float_array = wcet_array.astype(float)
    # replace all zeros by ones to perform the division
    path_wcet_array[path_wcet_array == 0.0] = 1.0
    # divide 2 n_islands x n_nodes float matrices and multiply it by a scalar int
    # path_wcet_array is not actually a n_islands x n_nodes matrix, but this is expanded 
    # automatically to match wcet_array shape
    rel_deadline_float_array    = (wcet_float_array / path_wcet_array) * dag_deadline
    rel_deadline_array = np.floor(rel_deadline_float_array).astype(int)
    if debug:
        print ('rel_deadline')
        print (rel_deadline_array)

    # The following check is equivalent to the following code:
    # for n in task_set:
    #     if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
    # meaning, it performs element wise comparison to check 
    # whether any task relative dealine is less than its wcet
    if np.less(rel_deadline_array,wcet_array).any():
        if debug:
            print ('WARNING: at least one task has wcet less than its relative deadline')
            print ('relative deadline:')
            print (rel_deadline_array)
            print ('wcet:')
            print (wcet_array)
        return False
    # copy the rel_deadline to the nodes
    for idx, row in enumerate(rel_deadline_array):
        task_set = np.argwhere(row>0.0).transpose()[0].tolist()
        for t in task_set: 
            G.nodes[t]["rel_deadline"] = rel_deadline_array[idx,t]

    ###########################
    # 4) calculate the PU utilization
    ###########################
    # The following code is equivalent to:
    # for i in islands:
    #     # remove the initial and last nodes of the DAG
    #     task_set = [t for t in i['placement'] if t != first_task and t != last_task]
    #     for t in task_set:
    #         # get the utilization for the current task t
    #         pu_utilization = (float(G.nodes[t]['wcet']) / float(G.nodes[t]['rel_deadline']))
    # replace all zeros by ones to perform the division
    rel_deadline_float_array[rel_deadline_float_array == 0.0] = 1.0
    task_utilization_array = wcet_float_array / rel_deadline_float_array
    if debug:
        print ('utilization')
        print (task_utilization_array)

    ###########################
    # 5) check the PU utilization constraint
    ###########################
    # 5.1) execute worst-fit task plament on PUs
    # 5.2) run minizinc when worst-fit fails 
    return check_utilization_mat(task_utilization_array)

# convert the task placement for all DAGs from the format of 'list of list' into numpy array
def convert_placement_list_to_np_array(dags,placement):
    n_tasks = sum([len(G.graph['actual_tasks']) for G in dags])
    placement_array = np.zeros((n_islands, n_tasks),dtype=int)
    for i in range(n_islands):
        for G in dags:
            for t in G.graph['actual_tasks']:
                # idx is required to support multi dags since the 2nd dag wont start with node 0
                idx = t - G.graph['first_task']
                if t in placement[i]:
                    placement_array[i,idx] = 1
                else:
                    placement_array[i,idx] = 0
    return placement_array

# np.set_printoptions(precision=2)
# n_nodes = len(G.nodes)
# # the 1st and last nodes are not in this list
# placement = [[8, 7, 6, 3, 1, 5], [], [2, 4]]
# placement_array = convert_placement_list_to_np_array(placement)
# feasible = check_placement_mat(placement_array)
# print (feasible)
# print ('wcet - rel_deadline')
# for t in range(len(G.nodes)):
#     print (G.nodes[t]["wcet"], G.nodes[t]["rel_deadline"])
# for i in islands:
#     print ('pu utilization:',i['pu_utilization'])
#     print ('pu placement',i['pu_placement'])
# sys.exit(1)

# Simplified algoritm used to prune some task placements
# out of the solution space
# Visiting in the reverse order to simplify the deletion from the list
# for l in range(search_space_size,0,-1):
#     if not check_placement_mat(leaf_list[l].islands):
#         print ('deleted placement', leaf_list[l].islands)
#         del(leaf_list[l])


# best_power = 999999.0
# best_freq = None
# islands[0]["placement"] = [9, 8, 7, 6, 3,  0, 1, 5]
# islands[1]["placement"] = []
# islands[2]["placement"] = [2, 4]
# for i in range(len(freq_seq)):
#    freqs_per_island_idx = freq_seq[i]
#    define_wcet()
#    feasible = define_rel_deadlines(G)
#    feasible2 = check_utilization()
#    power = define_power()
#    print ("{:.2f}".format(power), feasible, feasible2, freqs_per_island_idx)
#    if power < best_power and feasible and feasible2:
#        best_power = power
#        best_freq = list(freqs_per_island_idx)
# print ('BEST POWER:')
# print ("{:.2f}".format(best_power), best_freq)
# sys.exit(1)

# Pass the shared variables to the pool of processes.
# like shared values, locks cannot be shared in a Pool - instead, pass the 
# multiprocessing.Lock() at Pool creation time, using the initializer=init_globals.
# This will make your lock instance global in all the child workers.
# The init_globals is defined as a function - see init_globals() at the top.
# source : https://serveanswer.com/questions/multiprocessing-pool-map-multiple-arguments-with-shared-value-resolved
# source: https://stackoverflow.com/questions/53617425/sharing-a-counter-with-multiprocessing-pool
def init_globals(counter, l):
    global shared_best_power
    global shared_lock
    shared_best_power = counter
    shared_lock = l

# It converts an integer into a list of list representing the actual task mapping onto islands.
# Limitation: supports up to 32 tasks assuming a 32-bit int encodes the task mapping.
# It is based on a numerical system of base 'n_islands' which encodes a task mapping onto islands.
# For example:
# - Assuming 2 islands, i.e., a binary numerical system, the numbers:
#    - 00000:  says that all 5 tasks are mapped onto island 0
#    - 00001:  says that the 1st task is mapped onto island 1 and the rest onto island 0
#    - 00101:  says that the 1st and 3rd tasks are mapped onto island 1 and the rest onto island 0
# - Assuming 3 islands, i.e., a ternary numerical system, the numbers:
#    - 00000:  says that all 5 tasks are mapped onto island 0
#    - 00001:  says that 1st task is mapped onto island 1 and the rest onto island 0
#    - 00002:  says that 1st task is mapped onto island 2 and the rest onto island 0
#    - 00102:  says that 1st task is mapped onto island 2, the 3rd task onto island 1, and the rest onto island 0
def decode_mapping(curr_mapping, n_islands, n_tasks) -> list():
    mapping = [[] for i in range(n_islands)]
    island = 0
    # compute island onto which I'm mapping the i-th task
    for i in range(n_tasks): # scanning through all the n tasks
        island = int(curr_mapping / (n_islands**(i)) % n_islands)
        mapping[island].append(i)
    return mapping

# Return a task mapping onto islands from an integer code.
# This function is divided into two parts:
#  1) the mapping decoding based on a numerical system of 'n_islands' base
#  2) the mapping adjustment to discard the dummy tasks
def get_mapping(dags, curr_mapping, n_islands, n_tasks) -> list():
    # 1) the mapping decoding based on a numerical system of 'n_islands' base
    placement = decode_mapping(curr_mapping, n_islands, n_tasks)
    # 2) the mapping adjustment to discard the dummy tasks
    dummy_tasks = [dag.graph['first_task'] for dag in dags]
    dummy_tasks += [dag.graph['last_task'] for dag in dags]
    dummy_tasks.sort()
    # the idea is that, for each dummy task, increment all subsequent tasks
    # such that the final indexes in 'placement' points to the correct actual tasks,
    # skipping the dummy tasks
    for d in dummy_tasks:
        for i in placement:
            for idx,t in enumerate(i):
                if t >= d:
                    # # all subsequent indexes are incremented
                    i[idx] += 1
    # TODO place here some code to check the consistency of the generated placement
    # # cannot find a task in multiple island
    # for idx1,i1 in enumerate(placement[:-1]):
    #     for idx2, i2 in enumerate(placement[idx1+1:]):
    #         # set1 = set(i1['placement'])
    #         # set2 = set(i2['placement'])
    #         set1 = set(i1)
    #         set2 = set(i2)
    #         inter = set1.intersection(set2)
    #         if len(inter) > 0:
    #             print ('ERROR: task(s) ', inter, 'where found in islands',idx1,'and',idx2)
    #             sys.exit(1)
    return placement

# Main function that searches for the best placement found by this working process.
# Return format (best_power, best_task_placement, best_freq_idx).
# Return format (double, list of list, list)
def search_best_placement(placement_setup) -> tuple():
    print ('\nStarting work load:\n', ' - initial placement:', placement_setup[0], '\n  - # placements:', placement_setup[1], '\n  - process:', mp.current_process().name,mp.Process().name)
    current_placement = placement_setup[0]
    n_placements = placement_setup[1]
    # deepcopy the list of DAGs
    dags = []
    for i in placement_setup[2]:
        dags.append(i.copy())

    # count only the actual tasks
    n_tasks = sum([len(G.nodes) for G in dags]) - (len(dags)*2)

    # get the number of freq of each island
    n_freqs_per_island = [len(i['freqs']) for i in islands]
    # class that the encapsulate all the logic behind deciding the next frequecy sequence to be evaluated
    Fdag = freq_dag.Freq_DAG(n_freqs_per_island)

    # place holders for the best solution found by this process
    best_power = float("inf")
    best_task_placement = [0]*n_islands
    best_freq_idx = []

    # performance counters/timers
    execution_time_list = []
    time_to_write_shared_var = []
    time_to_read_shared_var = []
    evaluated_solutions = 0
    potential_solutions = 0
    best_solutions = 0
    bad_power = 0
    bad_deadline = 0
    bad_utilization = 0

    print("")
    for i in range(n_placements):
        # assume the following task placement onto the set of islands
        placement = get_mapping(dags, current_placement, n_islands, n_tasks)
        print (mp.current_process().name,':', placement)

        # Initialize freq to each island to their respective max freq.
        # The rational is that, if this task placement does not respect the DAG deadline
        # assigning their maximal frequencies, then this task placement cannot be a valid solution and
        # the search skip to the next task placement combination
        Fdag.set_task_placement(placement)

        if i%100 == 0:
            print ('Checking solution',i, 'out of',n_placements, 'possible mappings')
        keep_evaluating_freq_seq = True
        while keep_evaluating_freq_seq:
            # get the frequency sequence to be tested
            freqs_per_island_idx = Fdag.get()
            #print('freqs:',  hex(id(freqs_per_island_idx)), hex(id(Fdag)), freqs_per_island_idx)
            evaluated_solutions = evaluated_solutions +1
            start_time = time.time()
            if True:
                # define the wcet for each task based on which island each task is placed and the freq for each island
                define_wcet(dags, placement, freqs_per_island_idx)

                # wait for the updated power bound via the shared variable 
                # TODO no need to read it every iteration
                start_time2 = time.time()
                with shared_lock:
                    best_power = min(best_power, shared_best_power.value)
                time_to_read_shared_var.append(time.time() - start_time2)

                # check the power of this solution against the best power found among all working processes
                power = define_power(dags, placement, freqs_per_island_idx)
                if power >= best_power:
                    bad_power =bad_power +1
                    # freq_cnts[f] = freq_cnts[f] +1
                    Fdag.not_viable()
                    keep_evaluating_freq_seq = Fdag.next()
                    continue
                # if it reaches to this points, this is a so called 'potential solution'
                potential_solutions = potential_solutions +1
                # find the critical path and check whether the solution might be feasible.
                # If so, divide the dag deadline proportionly to the weight of each node in the critical path
                if not define_rel_deadlines(dags,freqs_per_island_idx): # TODO could have some variability in rel deadline assingment
                    bad_deadline =bad_deadline +1
                    # freq_cnts[f] = freq_cnts[f] +1
                    Fdag.not_viable()
                    keep_evaluating_freq_seq = Fdag.next()
                    continue
                # check the processor utilization constraint, i.e., each processor must have utilization <= 1.0
                # if not pu_utilization(0):
                if not check_utilization(dags, placement, freqs_per_island_idx):
                    bad_utilization =bad_utilization +1
                    # freq_cnts[f] = freq_cnts[f] +1
                    Fdag.not_viable()
                    keep_evaluating_freq_seq = Fdag.next()
                    continue
            else:
                start_time = time.time()
                placement = [i["placement"] for i in islands]
                placement_array = convert_placement_list_to_np_array(placement)
                feasible = check_placement_mat(placement_array)
                if not feasible:
                    bad_solutions =bad_solutions +1
                    # freq_cnts[f] = freq_cnts[f] +1
                    Fdag.not_viable()
                    keep_evaluating_freq_seq = Fdag.next()
                    continue
                elapsed_time = time.time() - start_time
                mat_time.append(elapsed_time)
            elapsed_time = time.time() - start_time
            execution_time_list.append(elapsed_time)
            # If it reached this point, then this a new best solution. 
            # Save this solution and shared the new power bound with the other working processes
            best_solutions = best_solutions +1
            best_power = power
            # save the best task placement onto the set of islands and the best frequency assignment
            best_task_placement = list(placement[:])
            best_freq_idx = list(freqs_per_island_idx)
            if debug:
                print ('solution found with power',"{:.2f}".format(best_power), best_task_placement, best_freq_idx)
                print ('WCET and REL DEADLINE:')
                for G in dags:
                    for n in G.graph['actual_tasks']:
                        print (n, G.nodes[n]["wcet"], G.nodes[n]["rel_deadline"])
            
            keep_evaluating_freq_seq = Fdag.next()
            # update the shared variable with the new power bound
            start_time = time.time()
            with shared_lock:
                shared_best_power.value = best_power
            elapsed_time = time.time() - start_time
            time_to_write_shared_var.append(elapsed_time)
        # reuse the frequency tree data structure for the next placement check
        Fdag.reinitiate_dag()
        # points to the next placement to be checked
        current_placement += 1

    print ('')
    print ('Performance counters/timers of process:', mp.current_process().name)
    print ('avg main execution time:',sum(execution_time_list)/float(n_placements))
    print ('avg time wating for reading the shared var:',sum(time_to_read_shared_var)/float(evaluated_solutions))
    if best_solutions > 0:
        print ('avg time wating for writing the shared var:',sum(time_to_write_shared_var)/float(best_solutions))
    
    print ('evaluated_solutions',evaluated_solutions)
    print ('potential_solutions',potential_solutions)
    print ('best_solutions', best_solutions)
    print ('bad_power', bad_power)
    print ('bad_deadline', bad_deadline)
    print ('bad_utilization', bad_utilization)
    print ('minizinc:', minizinc_cnt)

    # freq_cnts = Fdag.get_counters()
    # sum_freqs = sum([i[0]+i[1] for i in freq_cnts])
    # print ('freq histogram (unfeasible candidates):')
    # for i in range(len(freq_cnts)):
    #     if freq_cnts[i] != 0 :
    #         print ("{:.2f}".format(freq_cnts[i][0]/sum_freqs), ", {:.2f}".format(freq_cnts[i][1]/sum_freqs), freq_seq[i])

    print ('')
    if best_solutions > 0 :
        print ('Process:', mp.current_process().name, 'found a solution with power',"{:.2f}".format(best_power), best_task_placement, best_freq_idx)
        return (best_power, best_task_placement, best_freq_idx)
    else:
        print ('Process:', mp.current_process().name, 'was not able to find a better solution' )
        return tuple()

#####################
# GLOBALS
#####################
# variable representing the YAML with hw plarform definition
islands = None
# number of islands ... to avoid using len(islands) in the middle of the optim algo
n_islands = 0
# used to account the DAG precedence constraint into PU utilization calculation
# the first and last nodes are excluded
unrelated = None
# debug mode
debug = False
# hack to count the number of times minizinc is executed. need to update it to work w multiple procs
minizinc_cnt = 0

def main():
    global unrelated
    global islands
    global n_islands
    global debug

    args = argument_parsing()

    ################
    # Load Files
    ################
    # Read the software definition
    dags = load_sw(args.sw_file)
    # Used to account the DAG precedence constraint into PU utilization calculation
    unrelated = create_unrelated_task_set(dags)
    # Read the hardware platform definition
    islands = load_hw(args.hw_file)
    # number of islands ... to avoid using len(islands) in the middle of the optim algo
    n_islands = len(islands)

    ################
    # Program parameters
    ################
    # debug mode
    debug = args.debug
    # TODO: the result of the search must not change when the num of workers change 
    # graph/2dags-1task-each.yaml when changed from 1 to 3
    n_procs = args.processes
    # by default, run the entire search space. 
    max_placement_per_worker = args.iter

    ################
    # Parallel workload setup
    ################
    # subdivide the entire search space for parallel processing
    placement_setup_list = setup_workload(dags, n_procs, max_placement_per_worker)

    # shared variable used to keep the the processes updated related to the lowest power consumption found so far.
    # this is used to bound the search space
    manager = mp.Manager()
    # 'd' means double precision, 'i' is integer
    # 9 billions, 999 millions...
    shared_best_power = manager.Value('d', 9999999999.0)
    shared_lock = mp.Lock()
    # Alternatively, you could use Pool.imap_unordered, which starts returning 
    # results as soon as they are available instead of waiting until everything is finished. So you could tally the amount of returned results and use that to update the progress bar.
    # source: https://devdreamz.com/question/633149-how-to-use-values-in-a-multiprocessing-pool-with-python
    best_placement_list = []
    pool =  mp.Pool(initializer=init_globals, processes=n_procs, initargs=(shared_best_power,shared_lock,))
    # result_list = pool.map(search_best_placement, placement_setup_list)
    for best_placement in pool.map(search_best_placement, placement_setup_list):
        best_placement_list.append(best_placement)
    pool.close()
    pool.join()
    # remove the empty tuples, which means that that work package was not abble to find a suitable solution
    best_placement_list = [t for t in best_placement_list if t]

    # testing the shared variable
    # with shared_lock:
    #     print ('shared power:', shared_best_power.value)    

    ################
    # Print out the final results
    ################
    print("")
    if len(best_placement_list) > 0:
        best_placement_list.sort(key=lambda y: y[0])
        print ('The best solutions:')
        for idx,i in enumerate(best_placement_list):
            print('Solution:',idx)
            print (' - power:',"{:.2f}".format(i[0]))
            print (' - placement:', i[1])
            print (' - frequencies:',i[2])

        print("")
        print('Best solution:')
        print (' - power:',"{:.2f}".format(best_placement_list[0][0]))
        for i in range(n_islands):
            print (' - island:', i, ', capacity:', islands[i]['capacity'],', placement:', best_placement_list[0][1][i], ', frequency:',islands[i]['freqs'][best_placement_list[0][2][i]])
    else:
        print ('no feasiable solution was found :(')


if __name__ == '__main__':
    main()
