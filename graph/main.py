import networkx as nx
import pandas as pd


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
nodes_attrib = [1 for x in range(len(node_names))]

nodeData = pd.DataFrame({'name' : node_names,
                  'wcet' : wcet,
                  'wcet_ns' : wcet_ns,
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

print ('islands:')
for i in islands:
    print (i)

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

