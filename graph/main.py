import sys
from xmlrpc.client import boolean
import networkx as nx
import pandas as pd

# Ignored constraints:
# - affinity constraints
# - only one dag
# - no gpu or fpga

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
island = [0 for x in range(len(node_names))]
proc = [0 for x in range(len(node_names))]
nodes_attrib = [1 for x in range(len(node_names))]

nodeData = pd.DataFrame({'name' : node_names,
                  'wcet' : wcet,     # wcet at the reference freq
                  'wcet_ns' : wcet_ns,
                  'rel_deadline': rel_deadline,
                  'island': island,  # decision variables
                  'proc': proc,      # decision variables
                  'arrival_time': nodes_attrib,
                  'finish_time': nodes_attrib
                  })

G = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())

nx.set_node_attributes(G, nodeData.set_index('name').to_dict('index'))


G.graph['activation_period'] = 50
G.graph['deadline'] = 100

sparse_adj_mat = nx.adjacency_matrix(G)
array_adj_mat = sparse_adj_mat.toarray('C')
print (array_adj_mat)

islands = []
island1 = {}
island1['capacity'] = 1.0
island1['n_pus'] = 2
island1['busy_power'] = 100
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
    wcet_ns = G.nodes[t]['wcet_ns']
    wcet = G.nodes[t]['wcet']
    i = get_island(p)
    f_ref = max(islands[i]['freqs'])

    # get the index when the value == 1. get the freqs of an island i
    res = [x for x in range(len(freq_mat[i])) if freq_mat[i][x] == 1] 
    print (res)
    if len(res) != 1:
        print('ERROR: expecting size 1')
        sys.exit(0)
    f = islands[i]['freqs'][res[0]]
    capacity = islands[i]['capacity'] # TODO
    print (wcet_ns,wcet,capacity,f,f_ref)

    return wcet_ns + (capacity * (wcet-wcet_ns)/f * f_ref)

# new_wcet = calc_wcet(0,0,0)
# print (new_wcet)


# return whether the pu was overloaded or not
def pu_utilization(p) -> boolean:

    # get the index of the tasks deployed in PU p
    deployed_tasks = [x for x in range(len(deploy_mat)) if deploy_mat[x][p] == 1] 
    
    utilization = 0.0
    for t in deployed_tasks:
        new_wcet = calc_wcet(t,p,0)
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
# switching from the lowest freq to the 2nd lowest freq
freq_mat[0][1] = 0
freq_mat[0][2] = 1
print (pu_utilization(0))
print (pu_utilization(1))
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
# points to the island currently changing its freq
island_idx = 0
# points to the last island visited so far
last_visited_island_idx = 0
# number of islands ... to avoid using len(islands) in the middle of the optim algo
n_islands = len(islands)

# TODO
def create_minizinc_model():
    pass

#TODO
def run_minizinc()-> bool:
    return False

def inc_island_idx() -> bool:
    the_end = True
    inced = 0
    for i in range(n_islands):
        # if island idx i is pointing to its max freq
        if freqs_per_island_idx[i] < (n_freqs_per_island[i]-1):
            freqs_per_island_idx[i] = freqs_per_island_idx[i] +1
            inced = i
            break
        else:
            if i >= (n_islands-1):
                the_end = False
                return False
    
    for i in range(inced):
        freqs_per_island_idx[i] = 0
    return True

# running = True
# while running:
#     print ('searched freqs:')
#     for idx, i in enumerate(islands):
#         print (i['freqs'][freqs_per_island_idx[idx]])
#     running = inc_island_idx()

# sys.exit(1)

# stop running until a feasible solution is found or
# until all the island freqs where tested. In this case, it means no solution is possible
running = True
feasible = False
while running and not feasible:
    print ('current island:',island_idx, 'freq idx:', freqs_per_island_idx[island_idx])
    print ('searched freqs:')
    for idx, i in enumerate(islands):
        # print ('i:',idx)
        # print (freqs_per_island_idx)
        # print (freqs_per_island_idx[idx])
        print (i['freqs'][freqs_per_island_idx[idx]])
    print (freqs_per_island_idx)
    create_minizinc_model()
    feasible = run_minizinc()
    if not feasible:
        # increase the island freq and try the model again
        # stop when all the island are already at the max frequency
        running = inc_island_idx()

        # if the max freq of the current island was alerady tested, then go to the next island
        # and 
        if freqs_per_island_idx[island_idx] >= (n_freqs_per_island[island_idx]-1):
            if island_idx == last_visited_island_idx:
                if island_idx+1 >= n_islands:
                    # no solution
                    # no feasiable solution was found :(
                    # this means that the deadline is not feasible for the current set of tasks and processing units
                    # running = False
                    print ('no feasiable solution was found :(')
                    break
                else:
                    # try increasing the freq of the next island ...
                    last_visited_island_idx = last_visited_island_idx + 1
                    freqs_per_island_idx[last_visited_island_idx] = freqs_per_island_idx[last_visited_island_idx] + 1
                    # while the preceeding islands return to their respective minimal frequencies
                    for i in range(last_visited_island_idx):
                        freqs_per_island_idx[i] = 0
                    # and return to the initial island to start searching all over againg
                    island_idx = 0
            else:
                # try increasing the freq of the next island ...
                island_idx = island_idx + 1
                freqs_per_island_idx[island_idx] = freqs_per_island_idx[island_idx] + 1
                # while the preceeding islands return to their respective minimal frequencies
                for i in range(island_idx):
                    freqs_per_island_idx[i] = 0
                # and return to the initial island to start searching all over againg
                island_idx = 0
            # freqs_per_island_idx[island_idx] = freqs_per_island_idx[island_idx] + 1

            # # when all islands are on their max freq, then it's the end. Not solution was found
            # the_end = True
            # for i in range(n_islands):
            #     # if island i is pointing to its max freq
            #     if freqs_per_island_idx[i] < (n_freqs_per_island[i]-1):
            #         the_end = False
            #         break
            # # when the last island is already at its max 
            # print (the_end, island_idx)
            # if island_idx >= n_islands and the_end:
            #     # no feasiable solution was found :(
            #     # this means that the deadline is not feasible for the current set of tasks and processing units
            #     running = False
            #     print ('no feasiable solution was found :(')
            # else:
            #     # while the preceeding islands return to their respective minimal frequencies
            #     for i in range(island_idx):
            #         freqs_per_island_idx[i] = 0
            #     # and return to the initial island to start searching all over againg
        else:
            # try increasing the freq of the current island
            freqs_per_island_idx[island_idx] = freqs_per_island_idx[island_idx] + 1

    else:
        #solution found
        print ('solution found')
        running = False
