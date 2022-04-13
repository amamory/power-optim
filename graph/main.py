import sys
import os
import networkx as nx
import pandas as pd
import numpy as np
import math
import time
import yaml

# libs
import tree
import freq_dag




sw = None
with open('ex1-sw.yaml') as file:
    try:
        sw = yaml.safe_load(file)   
    except yaml.YAMLError as exc:
        print(exc)

# load the task definition
print ('TASKs:')
for idx, i in enumerate(sw['tasks']):
    print ('Task:',idx)
    print (' -',i)

# load the DAG definition
print ('')
print ('DAGs:')
G = None
for idx, i in enumerate(sw['dags']):
    print ('DAG:',idx)
    # DAG edges
    print (' - edges:',len(i['edge_list']))
    edge_list = []
    for e in i['edge_list']:
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
    print (' - ref_freq:',i['ref_freq'])
    G.graph['activation_period'] = i['activation_period']
    G.graph['deadline'] = i['deadline']
    # the reference freq is the freq used to characterize this application
    G.graph['ref_freq'] = i['ref_freq']
    # when it's deadling with multiple DAGs, this will find the first and last nodes of a DAG
    last_task = [t for t in G.nodes if len(list(G.successors(t))) == 0]
    first_task = [t for t in G.nodes if len(list(G.predecessors(t))) == 0]
    if len(first_task) != 1:
        print('ERROR: invalid DAG with multiple starting nodes')
        sys.exit(1)
    if len(last_task) != 1:
        print('ERROR: invalid DAG with multiple ending nodes')
        sys.exit(1)
    G.graph['first_task'] =  first_task[0]
    G.graph['last_task'] = last_task[0]


# load the hardware definition file
islands = None
with open('ex1-hw.yaml') as file:
    try:
        islands = yaml.safe_load(file)   
    except yaml.YAMLError as exc:
        print(exc)

# load power-related data from the CSV into the islands list of dict
print ('')
print ('ISLANDS:')
for i in islands['hw']:
    if not os.path.isfile(i['power_file']):
        print ('ERROR: file', i['power_file'], 'not found')
        sys.exit(1)
    data=pd.read_csv(i['power_file'])
    if  len(data.keys()) != 3 or data.keys()[0] != 'freq' or data.keys()[1] != 'busy_power_avg' or data.keys()[2] != 'idle_power_avg':
        print ('ERROR: file', i['power_file'], 'has an invalid syntax')
        sys.exit(1)
    # it is IMPORTANT that the power data is sorted in ascending frequency
    data.sort_values(by=['freq'], inplace=True)
    i['busy_power'] = list(data.busy_power_avg)
    i['idle_power'] = list(data.idle_power_avg)
    i['freqs'] = list(data.freq)
    i['placement'] = [] # the tasks assigned to this island
    i['pu_utilization'] = [0.0]*i['n_pus'] # the utilization on each PU
    i['pu_placement'] = [[] for aux in range(i['n_pus'])] # the tasks assigned to each PU
    del(i['power_file'])

for idx, i in enumerate(islands['hw']):
    print ('island:', idx)
    for key, value in i.items():
        print(" -", key, ":", value)

total_pus = 0
max_n_freq = 0
for i in islands['hw']:
    total_pus = total_pus + i['n_pus']
    max_n_freq = max(max_n_freq, len(i['freqs']))

# sort the islands by capacity. also drop the 'hw' level from it
islands = sorted(islands['hw'], key = lambda ele: ele['capacity'])

# get the number of freq of each island
n_freqs_per_island = [len(i['freqs']) for i in islands]

# number of islands ... to avoid using len(islands) in the middle of the optim algo
n_islands = len(islands)
# global index to the current freq in each island
# initialize them to the minimal freq, which is ALWAYS the first one
freqs_per_island_idx = [0]* n_islands

# debug mode
debug = False

# used to account the DAG precedence constraint into PU utilization calculation
unrelated = [
    [ 0 ], 
    [ 9 ], 
    [ 6, 7, 8 ], 
    [ 3, 6 ], 
    [ 5, 6, 8 ], 
    [ 1, 5 ], 
    [ 4, 5 ], 
    [ 1, 2, 3 ], 
    [ 3, 4 ]
]

print ("unrelated sets:")
for u in unrelated:
    print (u)

print ("dag properties:")
print (G.graph)

print ("node properties:")
for k,v in G.nodes(data=True):
    print(k,v)

# print ("edge properties:")
# for n1, n2, data in G.edges(data=True):
#     print(n1, n2, data)

# processing unit p,
def get_island(p):
    if p >= total_pus:
        print ('ERROR: invalid # of PUs', p)
        sys.exit(1)
    i_id = 0
    n = 0
    for i in islands:
        n = n + i['n_pus']
        if n > p:
            return i_id
        i_id = i_id +1
    
    print ('ERROR: should never reach here', p)
    sys.exit(1)

# return power consumed by the island i
def island_power(i) -> float:
    # get the index of the tasks deployed in island i
    deployed_tasks = [x for x in islands[i]['placement']] 
    
    # get the assigned freq and scales it down linearly with the the island max frequency
    # TODO read the power from a matrix
    # freq_scale_down = float(islands[i]['freqs'][freqs_per_island_idx[i]]) / float(islands[i]['freqs'][len(islands[i]['freqs'])-1])
    busy_power = islands[i]['busy_power'][freqs_per_island_idx[i]]
    idle_power = islands[i]['idle_power'][freqs_per_island_idx[i]]

    utilization = 0.0
    activation_period = G.graph['activation_period']
    for t in deployed_tasks:
        # assumes that wcet was calculated previously
        wcet = G.nodes[t]['wcet']
        utilization = utilization + (float(wcet)/float(activation_period))
    island_power = (islands[i]['n_pus'] * idle_power) + (busy_power-idle_power) * float(utilization)
    # print (i, utilization, island_power, deployed_tasks, freqs_per_island_idx)

    # TODO put a comment about (islands[i]['n_pus'] * idle_power)
    return island_power


# sum up the power of each island based on current task placement and island frequency
def define_power() ->  float:
    total_power = 0.0
    for i in range(n_islands):
        total_power = total_power +  island_power(i)
    return total_power

# TODO replace this linear search for a more 'binary search' approach, skiiping lots of unfeasible freqs combinations 
def create_frequency_sequence() -> list():
    freq_seq = []
    stop = False
    # start with all islands using their respective minimal frequencies
    freqs_per_island_idx = [0]*n_islands
    freq_seq.append(list(freqs_per_island_idx))
    while True:
        # points to the last incremented island
        inced = 0
        for i in range(n_islands):
            # if island idx i is not pointing to its max freq, then point to the next higher freq of this island
            if freqs_per_island_idx[i] < (n_freqs_per_island[i]-1):
                freqs_per_island_idx[i] = freqs_per_island_idx[i] +1
                inced = i
                break
            else:
                # if this is not the last island, then go to the next island to increment its freq
                if i >= (n_islands-1):
                    stop = True
        # all island before the last incremented island must start over at their lowest freq
        for i in range(inced):
            freqs_per_island_idx[i] = 0
        freq_seq.append(list(freqs_per_island_idx))
        # if all freqs are at their maximal value, then stop
        if stop:
            break
    # want to get have the maximal freq for each island 1st, to prune the search space faster
    reversed_freq_seq = list(reversed(freq_seq))
    return reversed_freq_seq

# define the wcet of each task based on in which island the task is placed
def define_wcet() -> None:
    for t in G.nodes:
        G.nodes[t]["wcet"] = 0
    # cannot find a task in multiple island
    for idx1,i1 in enumerate(islands[:-1]):
        for idx2, i2 in enumerate(islands[idx1+1:]):
            set1 = set(i1['placement'])
            set2 = set(i2['placement'])
            inter = set1.intersection(set2) 
            if len(inter) > 0:
                print ('ERROR: task(s) ', inter, 'where found in islands',idx1,'and',idx2)
                sys.exit(1)
    f_ref = G.graph['ref_freq']
    # calculate the wcet for each task
    # TODO make a matricial version of these loop, replacing scalar by matrices, eliminating the loops
    # prone to be executed in OpenMP/GPUs
    for idx, i in enumerate(islands):
        # get the frequency assigned to the island i
        f = i['freqs'][freqs_per_island_idx[idx]]
        capacity = i['capacity']
        for t in i['placement']:
            wcet_ref_ns = G.nodes[t]['wcet_ref_ns']
            wcet_ref = G.nodes[t]['wcet_ref']
            wcet = wcet_ref_ns + (capacity * (wcet_ref-wcet_ref_ns)/f * f_ref)
            G.nodes[t]["wcet"] = int(math.ceil(wcet))
    # cannot have a task not placed in an island
    for t in G.nodes:
        if G.nodes[t]["wcet"] == 0:
            print ('ERROR: wcet for task', t, 'not defined')
            sys.exit(1)

    # the 1st and last nodes have no computation
    G.nodes[G.graph['first_task']]["wcet"] = 0
    G.nodes[G.graph['last_task']]["wcet"] = 0

# assign relative deadline to each node
# potentially optmized version using 'shortest path' algorithms instead of 'all paths'
# compexity O(n*(n+e))
# return false if the deadline is not feasible
# TODO: code critical_path function based on the node weight rather than using shortest path based on edge weight
# https://stackoverflow.com/questions/6007289/calculating-the-critical-path-of-a-graph
def define_rel_deadlines2(G) -> bool:
    # main steps:   
    # 1) convert node weight into edge weight to find longest paths
    # 2) get the critical path to each node
    # 3) assign deadline to all nodes proportionally to its wcet and path wcet
    # 4) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    # 5) transfer the relative deadline back to the original DAG

    ####################
    # 1) convert node weight into edge weight to find longest paths
    ####################
    # deep copy the DAG
    H = G.copy()

    H.nodes[H.graph['first_task']]["wcet"] = 0
    H.nodes[H.graph['last_task']]["wcet"] = 0
    dag_deadline = H.graph['deadline']

    # get max node weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])
    max_task_wcet = max([H.nodes[n]["wcet"] for n in H.nodes])

    # if a single task is longer than the dag deadline, then this is not a solution
    if (max_task_wcet > dag_deadline):
        if debug:
            print ('WARNING: a single task wcet is longer than then DAG deadline', dag_deadline)
            for t in H.nodes:
                print (t, H.nodes[t]["wcet"])
        return False

    # assign the edge weight as the sum of the node weights
    for u, v, data in H.edges(data=True):
        # invert the edge weight since we are looking for the longest path
        data['weight'] = max_weight - (H.nodes[u]["wcet"] + H.nodes[v]["wcet"])

    for n in H.nodes:
        H.nodes[n]["rel_deadline"] = 0

    critical_path = nx.shortest_path(H,H.graph['first_task'],H.graph['last_task'],weight='weight')
    wcet_critical_path = sum([G.nodes[n1]["wcet"] for n1 in critical_path])
    if wcet_critical_path > dag_deadline:
        if debug:
            print ('WARNING: critical path', critical_path,'has lenght', wcet_critical_path, 'which is longer than then DAG deadline', dag_deadline)
        return False

    ####################
    # 2) get all paths to each end node
    ####################
    wcet_path_list  = [0]*len(H.nodes)
    # remove the initial and last nodes of the DAG
    task_set = [t for t in H.nodes if t != H.graph['first_task'] and t != H.graph['last_task']]
    for n in task_set:
        # get the critical path from the node 0 to the node n
        ipath = nx.shortest_path(H,0,n,weight='weight')
        isum = sum([H.nodes[n1]["wcet"] for n1 in ipath])
        # print (n, isum, ipath)
        # get the critical path from the node n to the last node 
        opath = nx.shortest_path(H,n,H.graph['last_task'], weight='weight')
        osum = sum([H.nodes[n1]["wcet"] for n1 in opath[1:]])
        # print (n, osum, opath)
        # assign the critical path to node n
        wcet_path_list[n] = isum+osum

    ############################
    # 3) assign deadline to all nodes proportionally to its wcet and path wcet
    ############################
    for n in task_set:
        wcet_ratio = float(H.nodes[n]["wcet"])/float(wcet_path_list[n])
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    

    ############################
    # 4) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    ############################
    # So far, it is not garanteed that the nodes have theirs respective maximal relative deadline. This last step does that 
    # by trying to increase the relative deadline of the last nodes.
    # get the last edges and nodes
    last_edges = H.in_edges(H.graph['last_task'])
    last_nodes = [e[0] for e in last_edges]
    for n in last_nodes:
        # the critical path
        path = nx.shortest_path(H,H.graph['first_task'],n,weight='weight')
        max_rel_deadline_sum = sum([H.nodes[n1]["rel_deadline"] for n1 in path])
        # assign any reamaning slack to its last node
        if max_rel_deadline_sum > dag_deadline:
            if debug:
                print('WARNING: path',path, 'has cost',max_rel_deadline_sum)
        H.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"] + (dag_deadline - max_rel_deadline_sum)
        if H.nodes[n]["rel_deadline"] <= 0:
            if debug:
                print('WARNING: path',path, 'has non positive cost',H.nodes[n]["rel_deadline"])

        # if the path is longer than the DAG deadline, this cannot be a solution
        if max_rel_deadline_sum > dag_deadline:
            if debug:
                print ('WARNING: path', path, 'takes', max_rel_deadline_sum,', longer than DAG deadline', dag_deadline)
            return False

    # the relative deadline of a task cannot be lower than its wcet
    for n in task_set:
        if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
            if debug:
                print ('WARNING: task', n, 'has wcet', H.nodes[n]["wcet"], 'and relative deadline', H.nodes[n]["rel_deadline"])
            return False
    ############################
    # 5) transfer the relative deadline back to the original DAG
    ############################
    for n in H.nodes:
        G.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"]

    return True

# assign relative deadline to each node
# TODO not scalable code with compexity O(n!)
# return false if the deadline is not feasible
# TODO it's also very inneficient because this is calculated for every frequency for every placement.
# the critical path and the relative deadlines could be pre-processed only once at startup
def define_rel_deadlines(G) -> bool:
    # main steps:   
    # 1) convert node weight into edge weight to find longest paths
    # 2) get all paths to each end node
    # 3) for each node, assign its max path wcet
    # 4) assign deadline to all nodes proportionally to its wcet and path wcet
    # 5) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    # 6) transfer the relative deadline back to the original DAG

    ####################
    # 1) convert node weight into edge weight to find longest paths
    ####################
    # deep copy the DAG
    H = G.copy()

    H.nodes[H.graph['first_task']]["wcet"] = 0
    H.nodes[H.graph['last_task']]["wcet"] = 0
    dag_deadline = H.graph['deadline']

    # get max node weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])
    start_time = time.time()
    max_task_wcet = max([H.nodes[n]["wcet"] for n in H.nodes])
    elapsed_time = time.time() - start_time

    # if a single task is longer than the dag deadline, then this is not a solution
    if (max_task_wcet > dag_deadline):
        if debug:
            print ('WARNING: a single task wcet is longer than then DAG deadline', dag_deadline)
            for t in H.nodes:
                print (t, H.nodes[t]["wcet"])
        temp_tuple = (terminate_counters[0][0]+1,terminate_counters[0][1]+elapsed_time)
        terminate_counters[0] = temp_tuple
        return False

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

    start_time = time.time()
    critical_path = nx.shortest_path(H,H.graph['first_task'],H.graph['last_task'],weight='weight')
    wcet_critical_path = sum([H.nodes[n1]["wcet"] for n1 in critical_path])
    elapsed_time = time.time() - start_time
    #print ('CRITICAL PATH:')
    #print (wcet_critical_path,critical_path)
    if wcet_critical_path > dag_deadline:
        if debug:
            print ('WARNING: critical path', critical_path,'has lenght', wcet_critical_path, 'which is longer than then DAG deadline', dag_deadline)
        temp_tuple = (terminate_counters[1][0]+1,terminate_counters[1][1]+elapsed_time)
        terminate_counters[1] = temp_tuple            
        return False

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
    # remove the initial and last nodes of the DAG
    task_set = [t for t in H.nodes if t != H.graph['first_task'] and t != H.graph['last_task']]
    for n in task_set:
        start_time = time.time()
        # get all the paths where node n is found
        paths_of_node_n = [p for p in all_paths_list if n in p]
        # get the wcet for each path
        path_wcet_sum = []
        for p in paths_of_node_n:
            # make a tuple w the sum of the path and the path
            path_wcet_sum.append((sum([H.nodes[n1]["wcet"] for n1 in p]), p))
        # get the path with the longest wcet
        max_partial_path = max(path_wcet_sum,key=lambda item:item[0])
        elapsed_time = time.time() - start_time
        # if a path was found longer than the DAG deadline, this cannot be a feasible solution
        if max_partial_path[0] > dag_deadline:
            if debug:
                print ('WARNING: critical path', max_partial_path[1], 'takes', max_partial_path[0],', longer than DAG deadline', dag_deadline)
            temp_tuple = (terminate_counters[2][0]+1,terminate_counters[2][1]+elapsed_time)
            terminate_counters[2] = temp_tuple
            #print ('PARTIAL PATH:')
            #for p in path_wcet_sum:
            #    print (p, [H.nodes[n1]["wcet"] for n1 in p[1]])
            #    p[1].append(9)
            #    for idx in range(len(p[1])-1):
            #        s = p[1][idx]
            #        t = p[1][idx+1]
            #        print(s,t,H.get_edge_data(s,t))
            #print ('CRITICAL PATH:')
            #print (wcet_critical_path, critical_path, [H.nodes[n1]["wcet"] for n1 in critical_path])    
            #sys.exit(1)            
            return False
        # save the longest of all paths
        if max_partial_path[0] > critical_path2[0]:
            critical_path2 = max_partial_path
        # mark each node of the selected path as 'path_wcet_list'
        for p in max_partial_path[1]:
            # replace node wcet by path wcet
            # so, each item in this list has its longest path wcet
            path_wcet_list[p] = max(max_partial_path[0],path_wcet_list[p])
    # print ('CRITICAL PATH:')
    # print (critical_path2)
    # if the critical path is longer than the DAG deadline, this cannot be a solution
    if critical_path2[0] > dag_deadline:
        # not expecting to reach this point unless there is a bug above
        print ('ERROR: critical path', critical_path2[1], 'takes', critical_path2[0],', longer than DAG deadline', dag_deadline)
        return False

    ############################
    # 4) assign deadline to all nodes proportionally to its wcet and path wcet
    ############################
    for n in task_set:
        wcet_ratio = float(H.nodes[n]["wcet"])/float(path_wcet_list[n])
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    

    ############################
    # 5) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
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
    for n in task_set:
        if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
            if debug:
                print ('WARNING: task', n, 'has wcet', H.nodes[n]["wcet"], 'and relative deadline', H.nodes[n]["rel_deadline"])
            temp_tuple = (terminate_counters[3][0]+1,0.0)
            terminate_counters[3] = temp_tuple
            return False

    ############################
    # 6) transfer the relative deadline back to the original DAG
    ############################
    for n in H.nodes:
        G.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"]

    return True

# Optimal solution to the place a task set onto PUs.
# It returns am empty list if it is indeed impossible to place the task set without breaking the utilization constraint.
# Otherwise, it returns the optimal task placement.
# TODO call minizinc and run Constraint Programming.
def optimal_placement(n_pus,task_set,graph) -> list:
    # create the minizinc dzn file
    # run minizinc
    # capture the task placement, if this is feasible
    return []

# It initially performs a worst_fit greedy approach to place the tasks of an island onto PUs.
# If the heuristic says that it is not possible to place the tasks, then it runs an exact optimal
# solver to confirm it. Hopefully, the heuristic will be sufficient most of the times.
# Returns false if any PU on any island exeeds the utilization threshold.
def check_utilization() -> bool:
    # for each island, run the placement heuristic and, if required, the exact solution
    temp_solution = []
    first_task = G.graph['first_task']
    last_task = G.graph['last_task']
    for i in islands:
        # to keep track of worst_fit heuristic with good data locality
        utilization_per_pu = [0.0]*i['n_pus']
        task_placement = [[] for aux in range(i['n_pus'])]
        # remove the initial and last nodes of the DAG
        task_set = [t for t in i['placement'] if t != first_task and t != last_task]
        for t in task_set:
            # get the PUs with minimal utilization
            pu = utilization_per_pu.index(min(utilization_per_pu))
            # get the utilization for the current task t
            pu_utilization = (float(G.nodes[t]['wcet']) / float(G.nodes[t]['rel_deadline']))        
            #print (t, pu, pu_utilization)
            # check if it is possible to assign this task to the pu, i.e., if the pu utilization is < 1.0
            if utilization_per_pu[pu] + pu_utilization > 1.0:
                # run minizinc to confirm whether it is indeed impossible to have this set of tasks placed on these PUs
                # TODO find a case where worst-fit does not find a solution but minizinc does
                task_placement = optimal_placement(i['n_pus'],i['placement'],G)
                if len(task_placement) == 0:
                    return False
                break
            else:
                utilization_per_pu[pu] = utilization_per_pu[pu] + pu_utilization
                task_placement[pu].append(t)
        temp_solution.append((utilization_per_pu,task_placement))
    # the solution is copied back to the islands data structure only if the task placement is feasible for all islands
    for idx, i in enumerate(islands):
        i['pu_utilization'] = list(temp_solution[idx][0])
        i['pu_placement'] = list(temp_solution[idx][1])
    
    return True

# define_rel_deadlines2(G)
# print ('NEW RELATIVE DEADLINES:')
# for n in G.nodes:
#     print (n, G.nodes[n]["rel_deadline"])

# define_rel_deadlines(G)
# print ('NEW RELATIVE DEADLINES2:')
# for n in G.nodes:
#     print (n, G.nodes[n]["rel_deadline"])
# sys.exit(1)

# This is a matrix-based (i.e. similar to matlab code) for cheking
# whether the task placement is viable, i.e., the critical path 
# does not exceed the DAG deadline and PU utilization is below 1.0
def check_placement_with_max_freq(placement_array) -> bool:
    # This procedure is divided into the following parts
    # 1) WCET definition for each task
    # 2) check the critical path 
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
    print ('wcet')
    print (wcet_array)
    sys.exit(1)


np.set_printoptions(precision=2)
n_nodes = len(G.nodes)
placement = [[9, 8, 7, 6,  0, 1, 5], [], [2, 3, 4]]
placement_array = np.zeros((n_islands, n_nodes),dtype=int)
for i in range(n_islands):
    for t in range(n_nodes):
        if t in placement[i]:
            placement_array[i,t] = 1
        else:
            placement_array[i,t] = 0
check_placement_with_max_freq(placement_array)
sys.exit(1)

# The number of combinations of t tasks in i islands
# is the number of leafs in a Perfect N-ary (i.e. i) Tree of height h (i.e. t).
# https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/5.04.N-ary_trees.pdf
# The number of nodes of a Perfect N-ary Tree of height h is: (N^(h+1)-1)/(N-1)
# Thus, the number of leafs in a Perfect N-ary Tree of height h is: ((N^(h+1)-1)/(N-1)) - ((N^(h)-1)/(N-1))
# Let a function C(i,t) denote the combinetion mentioned above, also decribed in the function
# tree.num_leafs_perfect_tree(ary,h):
#
#  - C(2,2) = 4
#  - C(2,3) = 8
#  - C(3,2) = 9
#  - C(3,3) = 27
#  - C(3,10) = 59,049 
#  - C(3,20) = 3,486,784,401 
#  - C(2,20) = 1,048,576 
#  - C(2,30) = 1,073,741,824 ==> assuming each node uses 1 byte of mem, which is obviously understimated, this tree would use at least 1Gbyte RAM
#
#  So, assuming C(2,30) = 1,073,741,824, and assuming each node uses 1 byte of mem, 
#  which is obviously understimated, this tree would use at least 1Gbyte RAM.
leaf_list = tree.task_island_combinations(n_islands,len(G.nodes))
# uncomment this to start the search with all tasks in the island with the biggest capacity
#leaf_list = list(reversed(leaf_list))
search_space_size = len(leaf_list)
# this is the sequence the set of frequencies must be evaluated
freq_seq = create_frequency_sequence()
# freq_cnts = [0]*len(freq_seq)
print ('Frequency sequences:', len(freq_seq))

# teminate search conditions used to understand the most prevalent ones
n_terminate_cond = 5
# a set of global counters used to gather stat data
terminate_counters = [(0,0.0)] * n_terminate_cond
terminate_counter_names = [
['task wcet > dag deadline'],
['critical path > dag deadline'],
['partial path > dag deadline'],
['task wcet > task rel deadline'],
['pu utilization violation']
]

# Simplified algoritm used to prune some task placements
# out of the solution space
# Visiting in the reverse order to simplify the deletion from the list
# for l in range(search_space_size,0,-1):
#     if not check_placement_with_max_freq(leaf_list[l].islands):
#         print ('deleted placement', leaf_list[l].islands)
#         del(leaf_list[l])

# for i in range(n_islands):
#    islands[i]["placement"] = leaf_list[0].islands[i]
# [[9, 8, 7, 6, 3,  0, 1, 5], [4], [2]]
islands[0]["placement"] = [9, 8, 7, 6, 3,  0, 1, 5]
islands[1]["placement"] = []
islands[2]["placement"] = [2, 4]
# islands[0]["placement"] = [9, 8, 7, 6, 4, 3, 2, 0, 1, 5]
# islands[1]["placement"] = []
# islands[2]["placement"] = []
freqs_per_island_idx = [2,2,2]
define_wcet()
# feasible = define_rel_deadlines(G)
# feasible2 = check_utilization()
# power = define_power()
# print ("{:.2f}".format(power), feasible, feasible2, freqs_per_island_idx)
for t in range(len(G.nodes)):
    print (G.nodes[t]["wcet"], G.nodes[t]["rel_deadline"])
sys.exit(1)

# best_power = 999999.0
# best_freq = None
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

# class that the encapsulate all the logic behind deciding the next frequecy sequence to be evaluated
Fdag = freq_dag.Freq_DAG(n_freqs_per_island)


best_power = float("inf")
best_task_placement = [0]*n_islands
best_freq_idx = []
l_idx = 0
evaluated_solutions = 0
potential_solutions = 0
best_solutions = 0
bad_solutions = 0
print("")
for l in leaf_list:
    # assume the following task placement onto the set of islands
    for i in range(n_islands):
        islands[i]["placement"] = l.islands[i]
    Fdag.set_task_placement(l.islands)
    # Initialize freq to each island to their respective max freq.
    # The rational is that, if this task placement does not respect the DAG deadline
    # assigning their maximal frequencies, then this task placement cannot be a valid solution and
    # the search skip to the next task placement combination
    if l_idx%100 == 0:
        print ('Checking solution',l_idx, 'out of',search_space_size, 'possible mappings')
    if l_idx >500:
        break
    # for f in range(len(freq_seq)):
    keep_evaluating_freq_seq = True
    while keep_evaluating_freq_seq:
        # get the frequency sequence to be tested
        #freqs_per_island_idx = freq_seq[f]
        freqs_per_island_idx = Fdag.get()
        # if debug:
        # print ('PLACEMENT and FREQs')
        # for i in range(n_islands):
        #     print(islands[i]["placement"])
        # print(freqs_per_island_idx)
        evaluated_solutions = evaluated_solutions +1
        # define the wcet for each task based on which island each task is placed and the freq for each island
        define_wcet()
        # find the critical path and check whether the solution might be feasible.
        # If so, divide the dag deadline proportionly to the weight of each node in the critical path
        if not define_rel_deadlines(G): # TODO could have some variability in rel deadline assingment
            bad_solutions =bad_solutions +1
            # freq_cnts[f] = freq_cnts[f] +1
            Fdag.not_viable()
            keep_evaluating_freq_seq = Fdag.next()
            continue
        # check the island/processor utilization feasibility
        # if not pu_utilization(0):
        if not check_utilization():
            bad_solutions =bad_solutions +1
            # freq_cnts[f] = freq_cnts[f] +1
            Fdag.not_viable()
            keep_evaluating_freq_seq = Fdag.next()
            continue
        # Since this solutions is feasible, check whether this was the lowest power found so far.
        # If so, update the best solution
        potential_solutions = potential_solutions +1
        power = define_power()
        if power < best_power:
            best_solutions = best_solutions +1
            best_power = power
            # save the best task placement onto the set of islands and the best frequency assignment
            for i in range(n_islands):
                best_task_placement[i] = list(l.islands[i])
            best_freq_idx = list(freqs_per_island_idx)
            print ('solution found with power',"{:.2f}".format(best_power), best_task_placement, best_freq_idx)
            if debug:
                print ('WCET and REL DEADLINE:')
                for n in G.nodes:
                    print (n, G.nodes[n]["wcet"], G.nodes[n]["rel_deadline"])
        
        keep_evaluating_freq_seq = Fdag.next()
    Fdag.reinitiate_dag()
    l_idx = l_idx +1

print("")
if best_solutions > 0:
    print ('solution found with power',"{:.2f}".format(best_power), best_task_placement, best_freq_idx)
else:
    print ('no feasiable solution was found :(')

print("")
print ('terminate counters:')
for idx,i in enumerate(terminate_counters):
    if i[0] == 0:
        print (terminate_counter_names[idx],i)
    else:
        print (terminate_counter_names[idx], i , i[1]/float(i[0]))

print("")
print ('total candidates evaluated', evaluated_solutions, 
    'bad candidates', bad_solutions, "({:.4f}%)".format(float(bad_solutions)/float(evaluated_solutions)),
    'potential solution', potential_solutions, "({:.4f}%)".format(float(potential_solutions)/float(evaluated_solutions)),
    'best solutions', best_solutions, "({:.4f}%)".format(float(best_solutions)/float(evaluated_solutions)))
freq_cnts = Fdag.get_counters()
sum_freqs = sum([i[0]+i[1] for i in freq_cnts])
print ('freq histogram (unfeasible candidates):')
for i in range(len(freq_cnts)):
    if freq_cnts[i] != 0 :
        print ("{:.2f}".format(freq_cnts[i][0]/sum_freqs), ", {:.2f}".format(freq_cnts[i][1]/sum_freqs), freq_seq[i])
