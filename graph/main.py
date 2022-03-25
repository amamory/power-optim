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

# y_s,m: init the freq assigned to a island
freq_mat = [ [0]*max_n_freq for i in range(len(islands))]
# assign each island to the minimal freq
for f in range(len(freq_mat)):
    freq_mat[f][0] = 1

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
    



# task i, processing unit p, operating performance points (freq) m
def calc_wcet(i,p,m) -> int:
    wcet_ns = G.nodes[i]['wcet_ns']
    wcet = G.nodes[i]['wcet']
    i = get_island(p)
    f_ref = max(islands[i]['freqs'])

    # get the index when the value == 1
    res = [x for x in range(len(freq_mat[i])) if freq_mat[i][x] == 1] 
    print (res)
    if len(res) != 1:
        print('ERROR: expecting size 1')
        sys.exit(0)
    f = islands[i]['freqs'][res[0]]
    capacity = 1.0 # TODO
    print (wcet_ns,wcet,capacity,f,f_ref)

    return wcet_ns + capacity * (wcet-wcet_ns)/f * f_ref

new_wcet = calc_wcet(0,0,0)
print (new_wcet)


# return whether the pu was overloaded or not
def pu_utilization(p) -> boolean:

    # get the index of the tasks deployed in PU p
    deployed_tasks = [x for x in range(len(deploy_mat)) if deploy_mat[x][p] == 1] 
    
    utilization = 0.0
    for t in deployed_tasks:
        new_wcet = calc_wcet(t,p,0)
        deadline = G.nodes[t]['rel_deadline']
        utilization = utilization + (float(new_wcet)/float(deadline))
    if utilization > 1.0:
        print ('WARNING: pu', p, 'has exeeded the utilization in', utilization)
        print (deployed_tasks, new_wcet, deadline, utilization)
        return False
    else:
        return True

pu_u = pu_utilization(0)
print (pu_u)

