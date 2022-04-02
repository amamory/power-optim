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
wcet_ns = [3 for x in range(len(node_names))]
rel_deadline = [x*3 for x in wcet]
# island = [0 for x in range(len(node_names))]
# proc = [0 for x in range(len(node_names))]
nodes_attrib = [1 for x in range(len(node_names))]
nodes_attrib[3] = 15

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
islands.append(island1)

island2 = {}
island2['capacity'] = 0.2
island2['n_pus'] = 2
island2['busy_power'] = 20
island2['idle_power'] = 2
island2['freqs'] = [50, 100, 200]
islands.append(island2)

island3 = {}
island3['capacity'] = 0.5
island3['n_pus'] = 2
island3['busy_power'] = 50
island3['idle_power'] = 5
island3['freqs'] = [100, 200, 300]
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

pu_u = pu_utilization(0)
print (pu_utilization(0))
print (pu_utilization(1))
# switching from the lowest freq to the 2nd lowest freq
freq_mat[0][0] = 0
freq_mat[0][1] = 1
print (pu_utilization(0))
print (pu_utilization(1))
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
    return True

# stop running until a feasible solution is found or
# until all the island freqs where tested. In this case, it means no solution is possible
running = True
feasible = False
# place all tasks in the lowest capacity island. Assuming the island were already sorted by capacity
task_placement = [0*len(node_names)]
# sort tasks in increasing wcet_ref
# the scalable and non scalable parts are added
wcet_ref_summed_up = [wcet[i]+wcet_ns[i] for i in range(len(node_names))]

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


wcet_list = [0]* len(node_names)

# def define_wcet():
#     wcet_list = []
#     for t in range(len(node_names)):
#         # TODO get its island
#         wcet = calc_wcet(t,i,0)
#         wcet_list.append(wcet)
#         H.nodes[u]["wcet"]


def define_rel_deadlines(G):
    # main steps:   
    # 1) find the critial path in the dag, i.e., the path with longest wcet for each node    
    # 2) assign deadline to all nodes in the critial path proportionally to its wcet
    # 3) assign the deadline for the reamaning nodes
    # 4) 2nd pass trying to find if it is possible to monotonicaly increase the "rel_deadline" of ant node without break the dag deadline
    # 5) transfer the relative deadline back to the original DAG

    ####################
    # 1) find the critial path in the dag, i.e., the path with longest wcet for each node    
    ####################
    
    # deep copy the DAG
    H = G.copy()

    # get max weight in the DAG. This is required to invert the weights since we need the longest path, not the shortest one
    max_weight = max([H.nodes[u]["wcet"] + H.nodes[v]["wcet"] for u, v in H.edges])
    print ("MAX:", max_weight)

    # assign the edge weight as the sum of the node weights
    for u, v, data in H.edges(data=True):
        # invert the edge weight since we are looking for the longest path
        data['weight'] = max_weight - (H.nodes[u]["wcet"] + H.nodes[v]["wcet"])

    critical_path = nx.dijkstra_path(H, 0, len(node_names)-1, weight='weight')

    # print (critical_path)
    # sys.exit(1)
    ####################
    # 2) assign deadline to all nodes in the critial path proportionally to its wcet
    ####################
    # put all rel_deadlines to zero so it's possible to know which nodes were already defined
    for n in H.nodes:
        H.nodes[n]["rel_deadline"] = 0
    # get the node weight sum  of the critical path
    sum_weight = sum([H.nodes[n]["wcet"] for n in critical_path])
    dag_deadline = G.graph['deadline']
    for n in critical_path:
        wcet_ratio = float(H.nodes[n]["wcet"])/float(sum_weight)
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.ceil(wcet_ratio*dag_deadline))
    # print('relative deadlines:')
    # for n in H.nodes:
    #     print (n, H.nodes[n]["rel_deadline"])
    # sys.exit(1)
    ####################
    # 3) assign the deadline for the reamaning nodes
    ####################
    # TODO: how bad is the performance of this function for larger DAGs ?!?! to be searched in the future
    # O(n!) in the complete graph of order n.
    all_paths_list = []
    all_paths = nx.all_simple_paths(H, 0, len(node_names)-1)
    # make it a list of paths
    all_paths_list.extend(all_paths)
    # print ('ALL PATHS:', len(all_paths_list))
    # for p in all_paths_list:
    #     print (p)
    # sys.exit(1)
    H.nodes[0]["wcet"] = 0
    H.nodes[len(H.nodes)-1]["wcet"] = 0
    # create a list of tuple of visited nodes with (node wcet,False)
    visited = [(H.nodes[n]["wcet"],False) for n in H.nodes]
    # a tuple to save the longest of all paths. format (path wcet, path)
    critical_path2 = (0,[])
    for n in range(1,len(H.nodes)-1):
        # if the ref_deadline is not defined yet
        # if H.nodes[n]["rel_deadline"] == 0:
        print ('ALL PATHS',n)
        # get all the paths where node n is found
        partial_paths = [p for p in all_paths_list if n in p]
        for p in partial_paths:
            print (p)
        
        # get the wcet for each path
        path_wcet_sum = []
        for p in partial_paths:
            # make a tuple w the sum of the path and the path
            path_wcet_sum.append((sum([H.nodes[n1]["wcet"] for n1 in p]), p))
        print ('ALL SUM PATHS',n)
        for p in path_wcet_sum:
            print (p)
        # get the path with the longest wcet
        max_partial_path = max(path_wcet_sum,key=lambda item:item[0])
        # save the longest of all paths
        if max_partial_path[0] > critical_path2[0]:
            critical_path2 = max_partial_path
        print ('MAX PATHS',n)
        print (max_partial_path)
        print (critical_path2)
        # mark each node of the selected path as 'visited'
        for p in max_partial_path[1]:
            # replace node wcet by path wcet
            # so, each item in this list has its longest path wcet
            visited[p] = (max(max_partial_path[0],visited[p][0]),True)
        # if all nodes were visited, break the loop. 'all' means logical 'and' of the list of booleans
        print ('visited:')
        print (visited)
        # if all([v[1] for v in visited]):
        #     print ('FIM!')
        #     break
    print ('CRITICAL PATH:')
    print (critical_path2)

    # H.nodes[0]["rel_deadline"] = 0
    # H.nodes[len(H.nodes)-1]["rel_deadline"] = 0
    for n in H.nodes:
        H.nodes[n]["rel_deadline"] = 0

    for n in range(1,len(H.nodes)-1):
        wcet_ratio = float(H.nodes[n]["wcet"])/float(visited[n][0])
        # assign rel_deadline proportional to its wcet
        H.nodes[n]["rel_deadline"] = int(math.floor(wcet_ratio*dag_deadline))    
    print('relative deadlines:', dag_deadline)

    ############################
    # 4) 2nd pass trying to find if it is possible to monotonicaly increase the "rel_deadline" of ant node without break the dag deadline
    ############################
    # So far, it is not garanteed that the nodes have theirs respective maximal relative deadline. This last step does that 
    # by trying to find if it is possible to monotonicaly increase the "rel_deadline" of ant node without break the dag deadline.
    # The strategy is to increase the relative deadline of the last nodes.
    print ('DESCENDENT')
    print (nx.descendents(H,len(H.nodes)-1))
    print (nx.ascendents(H,len(H.nodes)-1))

    ############################
    # 5) transfer the relative deadline back to the original DAG
    ############################
    for n in H.nodes:
        # print (n, H.nodes[n]["rel_deadline"], H.nodes[n]["wcet"], visited[n][0])
        print (n, H.nodes[n]["rel_deadline"])
        G.nodes[n]["rel_deadline"] = H.nodes[n]["rel_deadline"]
# 0 0
# 1 33
# 2 33
# 3 88
# 4 33
# 5 5
# 8 6  <=== correct would be 12
# 6 33
# 7 5
# 9 0
    sys.exit(1)
    # TODO to be continued
    pass

# sorted_tasks_by_wcet_ref = sort_task_and_return_idx(wcet_ref_summed_up)

define_rel_deadlines(G)

#print(sorted_tasks_by_wcet_ref)
sys.exit(1)
for t in range(len(node_names)):
    # initialize freq to each island to their respective minimal freq, which is ALWAYS the first one
    freqs_per_island_idx = [0]* len(islands)
    while running and not feasible:
        print ('searched freqs:')
        for idx, i in enumerate(islands):
            print (i['freqs'][freqs_per_island_idx[idx]])
        print (freqs_per_island_idx)
        # define_wcet()
        # find the critical path, divide the dag deadline proportionly to the wieght og each node in the critical path
        # define_rel_deadlines() # TODO could have some variability
        create_minizinc_model()
        feasible = run_minizinc()
        if not feasible:
            # increase the island freq and try the model again
            # stop when all the island are already at the max frequency
            running = inc_island_idx()
    # move task t to the next island

if not running:
    print ('no feasiable solution was found :(')