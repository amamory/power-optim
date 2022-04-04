import sys
from xmlrpc.client import boolean
import networkx as nx
import pandas as pd
import math

# Ignored constraints:
# - affinity constraints
# - update rel_deadline every time an island freq change
# - only one dag
# - precedence constraint is ignored
# - no gpu or fpga
# - how to get z ?

# % from: the leaving node for each edge
# % to: the entering node for each edge
# % list of edges indicating which nodes are connected
# e =  [| 1, 2
#       | 1, 3
#       | 2, 4
#       | 2, 5
#       | 3, 4
#       | 3, 5 |]; 

# % http://www.webgraphviz.com/
# % digraph G {
# %   "0" -> "1"
# %   "0" -> "2"
# %   "1" -> "3"
# %   "1" -> "4"
# %   "2" -> "3"
# %   "2" -> "4"
# % }


edge_list = [
    (0,1),
    (0,2),
    (0,3),
    (1,4),
    (2,4),
    (2,5),
    (3,5),
    (3,8),
    (4,6),
    (4,7),
    (4,8),
    (5,7),
    (6,9),
    (7,9),
    (8,9)
]

sources = [x[0] for x in edge_list]
targets = [x[1] for x in edge_list]
# edge attribute not used
weights = [1 for x in range(len(edge_list))]
print ('sources:',sources)
print ('targets:',targets)

linkData = pd.DataFrame({'source' : sources,
                  'target' : targets,
                  'weight' :weights})

# use the set to get unique node ids
node_names = {e for l in edge_list for e in l}
# then get ride of the set and transform it into a array
node_names = list(node_names)
print ('node names:',node_names)
wcet = [10 for x in range(len(node_names))]
wcet[3] = 15
wcet_ns = [3 for x in range(len(node_names))]
rel_deadline = [x*3 for x in wcet]
# island = [0 for x in range(len(node_names))]
# proc = [0 for x in range(len(node_names))]
nodes_attrib = [1 for x in range(len(node_names))]

nodeData = pd.DataFrame({'name' : node_names,
                  'wcet_ref' : wcet,     # wcet at the reference freq
                  'wcet_ref_ns' : wcet_ns,
                  'rel_deadline': rel_deadline,
                  'wcet': nodes_attrib # decision variables
                #   'island': island,  # decision variables
                #   'proc': proc,      # decision variables
                #   'arrival_time': nodes_attrib,
                #   'finish_time': nodes_attrib
                  })

G = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())

nx.set_node_attributes(G, nodeData.set_index('name').to_dict('index'))


G.graph['activation_period'] = 100
G.graph['deadline'] = 100
# the reference freq is the freq of the 1st island at its highest freq
# considering which island ?!?!? let us say it's the 1st island of the list of islands
G.graph['ref_freq'] = 1000

sparse_adj_mat = nx.adjacency_matrix(G)
array_adj_mat = sparse_adj_mat.toarray('C')
print (array_adj_mat)

islands = []
island1 = {}
island1['capacity'] = 1.0
island1['n_pus'] = 2
island1['busy_power'] = 100 # TODO review grabriele's paper to get the funtion of power compared to freq
island1['idle_power'] = 20
island1['freqs'] = [100, 500, 1000]
island1['placement'] = [] # the tasks placed in this island
islands.append(island1)

island2 = {}
island2['capacity'] = 0.2
island2['n_pus'] = 2
island2['busy_power'] = 20
island2['idle_power'] = 2
island2['freqs'] = [50, 100, 200]
island2['placement'] = [] # the tasks placed in this island
islands.append(island2)

island3 = {}
island3['capacity'] = 0.5
island3['n_pus'] = 2
island3['busy_power'] = 50
island3['idle_power'] = 5
island3['freqs'] = [100, 200, 300]
island3['placement'] = [] # the tasks placed in this island
islands.append(island3)


total_pus = 0
max_n_freq = 0
for i in islands:
    total_pus = total_pus + i['n_pus']
    max_n_freq = max(max_n_freq, len(i['freqs']))

print ('islands:')
for i in islands:
    print (i)

# x_i,u: init proc deploy matrix w zeros
deploy_mat = [ [0]*total_pus for i in range(len(node_names))]
# assign t0 to p0, t1 to p1, etc
pu = 0
for t in range(len(deploy_mat)):
    deploy_mat[t][pu] = 1
    pu = (pu + 1) % total_pus

print ('deploy_mat:')
for i in range(len(deploy_mat)):
    print (deploy_mat[i])

# y_s,m: init the freq assigned to a island
freq_mat = [ [0]*max_n_freq for i in range(len(islands))]
# assign each island to the minimal freq
for f in range(len(freq_mat)):
    freq_mat[f][0] = 1

print ('freq_mat:')
for i in range(len(freq_mat)):
    print (freq_mat[i])

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

print ("edge properties:")
for n1, n2, data in G.edges(data=True):
    print(n1, n2, data)

# processing unit p,
def get_island(p):
    if p >= total_pus:
        print ('ERROR: invalid # of PUs', p)
        sys.exit(0)
    i_id = 0
    n = 0
    for i in islands:
        n = n + i['n_pus']
        if n > p:
            return i_id
        i_id = i_id +1
    
    print ('ERROR: should never reach here', p)
    sys.exit(0)


# task t, processing unit p, operating performance points (freq) m
def calc_wcet(t,p,m) -> int:
    wcet_ns = G.nodes[t]['wcet_ref_ns']
    wcet = G.nodes[t]['wcet_ref']
    i = get_island(p)
    # that's somehow arbitrary definition
    f_ref = G.graph['ref_freq']

    # get the index when the value == 1. get the freqs of an island i
    res = [x for x in range(len(freq_mat[i])) if freq_mat[i][x] == 1] 
    #print (res)
    if len(res) != 1:
        print('ERROR: expecting size 1')
        sys.exit(0)
    f = islands[i]['freqs'][res[0]]
    capacity = islands[i]['capacity']
    #print (wcet_ns,wcet,capacity,f,f_ref)

    return wcet_ns + (capacity * (wcet-wcet_ns)/f * f_ref)

for i in range(len(node_names)):
    wcets = [
            int(calc_wcet(i,0,0)), int(calc_wcet(i,0,0)), 
            int(calc_wcet(i,2,0)), int(calc_wcet(i,2,0)), 
            int(calc_wcet(i,4,0)), int(calc_wcet(i,4,0))
            ]
    print ('task:', i , 'wcets:', wcets)
#sys.exit(1)

# return whether the pu was overloaded or not
def pu_utilization(p) -> boolean:

    # get the index of the tasks deployed in PU p
    deployed_tasks = [x for x in range(len(deploy_mat)) if deploy_mat[x][p] == 1] 
    
    utilization = 0.0
    for t in deployed_tasks:
        new_wcet = calc_wcet(t,p,0)
        # TODO rel_deadline must be updated every time an island freq change
        deadline = G.nodes[t]['rel_deadline']
        utilization = utilization + (float(new_wcet)/float(deadline))
    print (deployed_tasks, new_wcet, deadline, utilization)
    if utilization > 1.0:
        print ('WARNING: pu', p, 'has exeeded the utilization in', utilization)
        return False
    else:
        return True

# pu_u = pu_utilization(0)
# print (pu_utilization(0))
# print (pu_utilization(1))
# # switching from the lowest freq to the 2nd lowest freq
# freq_mat[0][0] = 0
# freq_mat[0][1] = 1
# print (pu_utilization(0))
# print (pu_utilization(1))
# # switching from the lowest freq to the 2nd lowest freq
# freq_mat[0][1] = 0
# freq_mat[0][2] = 1
# print (pu_utilization(0))
# print (pu_utilization(1))
# the 
print (pu_utilization(2))
print (pu_utilization(3))



# return power consumed by the pu p
def pu_power(p) -> float:
    # get the index of the tasks deployed in PU p
    deployed_tasks = [x for x in range(len(deploy_mat)) if deploy_mat[x][p] == 1] 
    
    i = get_island(p)
    print ('p:',p,i)
    busy_power = islands[i]['busy_power']
    idle_power = islands[i]['idle_power']

    utilization = 0.0
    z=1
    activation_period = G.graph['activation_period']
    for t in deployed_tasks:
        new_wcet = calc_wcet(t,p,0)
        # TODO get z
        utilization = utilization + (z*float(new_wcet)/float(activation_period))
    print (deployed_tasks, new_wcet, z, activation_period, utilization)

    return idle_power + (busy_power-idle_power) * utilization

for p in range(total_pus):
    print ('power:', pu_power(p))

#
# Heusristic 
#

# sort the islands by idle_power
islands = sorted(islands, key = lambda ele: ele['idle_power'])

# print ('islands:')
# for i in islands:
#     print (i)

# assign each island to the minimal freq
for f in range(len(freq_mat)):
    freq_mat[f][0] = 1

# get the number of freq of each island
n_freqs_per_island = [len(i['freqs']) for i in islands]

# index to the current freq in each island
# initialize them to the minimal freq, which is ALWAYS the first one
freqs_per_island_idx = [0]* len(islands)
# number of islands ... to avoid using len(islands) in the middle of the optim algo
n_islands = len(islands)

# TODO
def create_minizinc_model():
    pass

# TODO
def run_minizinc()-> bool:
    return False

# TODO replace this linear search for a more 'binary search' approach, skiiping lots of unfeasible freqs combinations 
def inc_island_idx() -> bool:
    # points to the last incremented island
    inced = 0
    for i in range(n_islands):
        # if there is no task placed in this island, skip it if this is not the last island
        if len(islands[i]['placement'])==0:
            if i == n_islands-1:
                return False
            else:
                continue
        # if island idx i is not pointing to its max freq, then point to the next higher freq of this island
        if freqs_per_island_idx[i] < (n_freqs_per_island[i]-1):
            freqs_per_island_idx[i] = freqs_per_island_idx[i] +1
            inced = i
            break
        else:
            # if this is not the last island, then go to the next island to increment its freq
            if i >= (n_islands-1):
                return False
                
    # all island before the last incremented island must start over at their lowest freq
    for i in range(inced):
        freqs_per_island_idx[i] = 0

    # assign the selected freq to the binary matrix
    freq_mat = [ [0]*max_n_freq for i in range(len(islands))]
    for i in range(n_islands):
        freq_mat[i][freqs_per_island_idx[i]] = 1

    return True

# take the first element for sort
def take_first(elem):
    return elem[0]

# get the task indexes such that these indexes are sorted by wcet_ref
# for instance, [3,4,2,5,2,6,1] returns [6,2,4,0,1,3,5]
def sort_task_and_return_idx(wcet):
    # add the 'visited' tag to each value by creating a touple (int,bool)
    wcet_tagged = [(wcet[i],False) for i in range(len(wcet))]
    wcet_sorted = sorted(wcet_tagged,key=take_first)
    # the result to be returned
    sorted_idx = []
    while(len(wcet_sorted)>0):
        # get the position where the lowest value can be found
        idx = -1
        for i in range(len(wcet_tagged)):
            if wcet_tagged[i][0] == wcet_sorted[0][0] and not wcet_tagged[i][1]:
                idx = i
                # mark as visited before
                wcet_tagged[i] = (wcet_tagged[i][0],True)
                break
        if idx < 0:
            print ("ERROR: check the function 'sort_task_and_return_idx'")
            sys.exit(1)
        sorted_idx.append(idx)
        # delete the 1st node and continue until this list is empty
        wcet_sorted = wcet_sorted[1:]
    
    return sorted_idx

# testing = [3,4,2,5,2,6,1]
# testing_sorted = sort_task_and_return_idx(testing)
# print (testing)
# print (testing_sorted)

# define the wcet of each task based on in which island the task is placed
def define_wcet():
    for t in G.nodes:
        G.nodes[t]["wcet"] = 0
    # cannot find a task in multiple island
    for idx1,i1 in enumerate(islands[:-1]):
        for idx2, i2 in enumerate(islands[idx1+1:]):
            set1 = set(i1['placement'])
            set2 = set(i2['placement'])
            inter = set1.intersection(set2) 
            if len(inter) > 0:
                print ('WARNING: task(s) ', inter, 'where fond in islands',idx1,'and',idx2)
    # get a valid processing unit id for each island
    proc_id = []
    n_pu = 0
    for i in islands:
        proc_id.append(n_pu)
        n_pu = n_pu + i['n_pus'] 
    # calculate the wcet for each task
    for idx,i in enumerate(islands):
        for t in i['placement']:
            wcet = calc_wcet(t,proc_id[idx],0)
            G.nodes[t]["wcet"] = int(math.ceil(wcet))
    # cannot have a task not placed in an island
    for t in G.nodes:
        if G.nodes[t]["wcet"] == 0:
            print ('WARNING: wcet for task', t, 'not defined')

# assign relative deadline to each node
# potentially optmized version using 'shortest path' algorithms instead of 'all paths'
# compexity O(n*(n+v))
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

    last_node = len(H.nodes)-1
    H.nodes[0]["wcet"] = 0
    H.nodes[last_node]["wcet"] = 0
    dag_deadline = H.graph['deadline']

    # get max node weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])

    # if a single task is longer than the dag deadline, then this is not a solution
    if (max_weight > dag_deadline):
        print ('WARNINIG: a single task wcet is longer than then DAG deadline', dag_deadline)
        for t in H.nodes:
            print (t, H.nodes[t]["wcet"])
        return False

    # assign the edge weight as the sum of the node weights
    for u, v, data in H.edges(data=True):
        # invert the edge weight since we are looking for the longest path
        data['weight'] = max_weight - (H.nodes[u]["wcet"] + H.nodes[v]["wcet"])

    for n in H.nodes:
        H.nodes[n]["rel_deadline"] = 0

    ####################
    # 2) get all paths to each end node
    ####################
    wcet_path_list  = [0]*len(H.nodes)
    for n in range(1,last_node):
        # get the critical path from the node 0 to the node n
        ipath = nx.shortest_path(H,0,n,weight='weight')
        isum = sum([H.nodes[n1]["wcet"] for n1 in ipath])
        # print (n, isum, ipath)
        # get the critical path from the node n to the last node 
        opath = nx.shortest_path(H,n,last_node, weight='weight')
        osum = sum([H.nodes[n1]["wcet"] for n1 in opath[1:]])
        # print (n, osum, opath)
        # assign the critical path to node n
        wcet_path_list[n] = isum+osum

    ############################
    # 3) assign deadline to all nodes proportionally to its wcet and path wcet
    ############################
    for n in range(1,len(H.nodes)-1):
        wcet_ratio = float(H.nodes[n]["wcet"])/float(wcet_path_list[n])
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    

    ############################
    # 4) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    ############################
    # So far, it is not garanteed that the nodes have theirs respective maximal relative deadline. This last step does that 
    # by trying to increase the relative deadline of the last nodes.
    # get the last edges and nodes
    last_edges = H.in_edges(len(H.nodes)-1)
    last_nodes = [e[0] for e in last_edges]
    for n in last_nodes:
        # the critical path
        path = nx.shortest_path(H,0,n,weight='weight')
        max_rel_deadline_sum = sum([H.nodes[n1]["rel_deadline"] for n1 in path])
        # assign any reamaning slack to its last node
        if max_rel_deadline_sum > dag_deadline:
            print('WARNING: path',path, 'has cost',max_rel_deadline_sum)
        H.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"] + (dag_deadline - max_rel_deadline_sum)
        if H.nodes[n]["rel_deadline"] <= 0:
            print('WARNING: path',path, 'has non positive cost',H.nodes[n]["rel_deadline"])

        # if the path is longer than the DAG deadline, this cannot be a solution
        if max_rel_deadline_sum > dag_deadline:
            print ('WARNING: path', path, 'takes', max_rel_deadline_sum,', longer than DAG deadline', dag_deadline)
            return False

    # the relative deadline of a task cannot be lower than its wcet
    for n in H.nodes:
        if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
            print ('WARNING: task', n, 'has wcet', H.nodes[n]["wcet"], 'and relative deadline', H.nodes[n]["rel_deadline"])
            return False
    ############################
    # 5) transfer the relative deadline back to the original DAG
    ############################
    for n in H.nodes:
        G.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"]

    return True

# assign relative deadline to each node
# not scalable code with compexity O(n!)
# return false if the deadline is not feasible
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

    last_node = len(H.nodes)-1
    H.nodes[0]["wcet"] = 0
    H.nodes[last_node]["wcet"] = 0
    dag_deadline = H.graph['deadline']

    # get max node weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])
    # print ("MAX:", max_weight)

    # if a single task is longer than the dag deadline, then this is not a solution
    if (max_weight > dag_deadline):
        print ('WARNINIG: a single task wcet is longer than then DAG deadline', dag_deadline)
        for t in H.nodes:
            print (t, H.nodes[t]["wcet"])
        return False

    # assign the edge weight as the sum of the node weights
    for u, v, data in H.edges(data=True):
        # invert the edge weight since we are looking for the longest path
        data['weight'] = max_weight - (H.nodes[u]["wcet"] + H.nodes[v]["wcet"])

    ####################
    # 2) get all paths to each end node
    ####################
    # get the last edges and nodes
    last_edges = H.in_edges(len(H.nodes)-1)
    last_nodes = [e[0] for e in last_edges]
    # get the paths to the last nodes
    # TODO: bad scalability !!! the 'all_simple_paths' function is O(n!) in the complete graph of order n.
    paths_from_last_nodes = []
    all_paths_list = []
    for n in last_nodes:
        path_list = []
        # get all the paths to last node n
        paths = nx.all_simple_paths(H, 0, n)
        # make it a list of paths
        path_list.extend(paths)
        paths_from_last_nodes.append(path_list)
        all_paths_list = all_paths_list + path_list

    # print ('ALL PATHS:', len(all_paths_list))
    # for p in all_paths_list:
    #     print (p)
    # sys.exit(1)
    ####################
    # 3) for each node, assign its max path wcet
    ####################
    H.nodes[0]["wcet"] = 0
    H.nodes[len(H.nodes)-1]["wcet"] = 0
    # create a list with node wcet
    wcet_list = [H.nodes[n]["wcet"] for n in H.nodes]
    # a tuple to save the longest of all paths. format (path wcet, path)
    critical_path2 = (0,[])
    for n in range(1,len(H.nodes)-1):
        # print ('ALL PATHS',n)
        # get all the paths where node n is found
        partial_paths = [p for p in all_paths_list if n in p]
        # for p in partial_paths:
        #     print (p)
        # get the wcet for each path
        path_wcet_sum = []
        for p in partial_paths:
            # make a tuple w the sum of the path and the path
            path_wcet_sum.append((sum([H.nodes[n1]["wcet"] for n1 in p]), p))
        # print ('ALL SUM PATHS',n)
        # for p in path_wcet_sum:
        #     print (p)
        # get the path with the longest wcet
        max_partial_path = max(path_wcet_sum,key=lambda item:item[0])
        # save the longest of all paths
        if max_partial_path[0] > critical_path2[0]:
            critical_path2 = max_partial_path
        # print ('MAX PATHS',n)
        # print (max_partial_path)
        # print (critical_path2)
        # mark each node of the selected path as 'wcet_list'
        for p in max_partial_path[1]:
            # replace node wcet by path wcet
            # so, each item in this list has its longest path wcet
            wcet_list[p] = max(max_partial_path[0],wcet_list[p])
        # print ('wcet_list:')
        # print (wcet_list)
    print ('CRITICAL PATH:')
    print (critical_path2)

    ############################
    # 4) assign deadline to all nodes proportionally to its wcet and path wcet
    ############################
    for n in range(1,len(H.nodes)-1):
        wcet_ratio = float(H.nodes[n]["wcet"])/float(wcet_list[n])
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    

    ############################
    # 5) 2nd pass of trying to increase the "rel_deadline" of the last nodes without break the dag deadline
    ############################
    # So far, it is not garanteed that the nodes have theirs respective maximal relative deadline. This last step does that 
    # by trying to increase the relative deadline of the last nodes.
    for n in paths_from_last_nodes:
        # the last node index
        node = n[0][-1]
        rel_deadline_sum = []
        # print ('PATHS:', node)
        for p in n:
            # print (p)
            # make a tuple w the sum of the path and the path
            rel_deadline_sum.append(sum([H.nodes[n1]["rel_deadline"] for n1 in p]))
        # get the path with the longest sum of rel_deadline
        max_rel_deadline_sum = max(rel_deadline_sum)
        # assign any reamaning slack to its last node
        H.nodes[node]["rel_deadline"] = H.nodes[node]["rel_deadline"] + (dag_deadline - max_rel_deadline_sum)

    # if the critical path is longer than the DAG deadline, this cannot be a solution
    if critical_path2[0] > dag_deadline:
        print ('WARNING: critical path', critical_path2[1], 'takes', critical_path2[0],', longer than DAG deadline', dag_deadline)
        return False

    # the relative deadline of a task cannot be lower than its wcet
    for n in H.nodes:
        if H.nodes[n]["rel_deadline"] < H.nodes[n]["wcet"]:
            print ('WARNING: task', n, 'has wcet', H.nodes[n]["wcet"], 'and relative deadline', H.nodes[n]["rel_deadline"])
            return False

    ############################
    # 6) transfer the relative deadline back to the original DAG
    ############################
    for n in H.nodes:
        #print (n, H.nodes[n]["rel_deadline"])
        G.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"]

    return True

define_rel_deadlines2(G)
print ('NEW RELATIVE DEADLINES:')
for n in G.nodes:
    print (n, G.nodes[n]["rel_deadline"])

define_rel_deadlines(G)
print ('NEW RELATIVE DEADLINES:')
for n in G.nodes:
    print (n, G.nodes[n]["rel_deadline"])

define_wcet()
print ('WCET:')
for n in G.nodes:
    print (n, G.nodes[n]["wcet"])

# return false if task t was already in the last island
def move_task_between_islands(t) -> bool:
    found = False
    for i in range(n_islands):
        if t in islands[i]['placement']:
            if i == n_islands-1:
                return False
            else:
                islands[i]['placement'].remove(t)
                islands[i+1]['placement'].append(t)
                found = True
                break
    if not found:
        print('WARNING: task',t, 'not assigned to any island')
    return True


# stop running until a feasible solution is found or
# until all the island freqs where tested. In this case, it means no solution is possible
running = True
feasible = False
# sort tasks in increasing wcet_ref
# the scalable and non scalable parts are added
# wcet_ref_summed_up = [wcet[i]+wcet_ns[i] for i in range(len(node_names))]
# sorted_tasks_by_wcet_ref = sort_task_and_return_idx(wcet_ref_summed_up)
# print(sorted_tasks_by_wcet_ref)
# sys.exit(1)

# place all tasks into the 1st island, the one with lowest capacity
islands[0]['placement'] = range(len(node_names))

# The number of combinations of t tasks in i islands
# is the number of leafs in a Perfect N-ary (i.e. i) Tree of height h (i.e. t).
# The number of nodes of a Perfect N-ary Tree of height h is: (N^(h+1)-1)/(N-1)
# Thus, the number of leafs in a Perfect N-ary Tree of height h is: ((N^(h+1)-1)/(N-1)) - ((N^(h)-1)/(N-1))
# Let a function C(i,t) denote the combinetion mentioned above. 
#  - C(2,2) = 4
#  - C(2,3) = 8
#  - C(3,2) = 9
#  - C(3,10) = 59,049 
#  - C(3,20) = 3,486,784,401 
#  - C(2,20) = 1,048,576 
#  - C(2,30) = 1,073,741,824 

for t in range(len(node_names)):
    # initialize freq to each island to their respective minimal freq, which is ALWAYS the first one
    freqs_per_island_idx = [0]* len(islands)
    while running and not feasible:
        print ('searched freqs:')
        for idx, i in enumerate(islands):
            print (i['freqs'][freqs_per_island_idx[idx]])
        print (freqs_per_island_idx)
        # define the wcet for each task based on which island each task is placed and the freq for each island
        define_wcet()
        # find the critical path, divide the dag deadline proportionly to the wieght og each node in the critical path
        if define_rel_deadlines(G): # TODO could have some variability
            create_minizinc_model()
            feasible = run_minizinc()
        else:
            print ('not a solution:')
            feasible = False
        print ('WCET and REL DEADLINE:')
        for n in G.nodes:
            print (n, G.nodes[n]["wcet"], G.nodes[n]["rel_deadline"])
        # sys.exit(1)
        if not feasible:
            # increase the island freq and try the model again
            # stop when all the island are already at the max frequency
            running = inc_island_idx()
    # move task t to the next island
    move_task_between_islands(t)

if not running:
    print ('no feasiable solution was found :(')