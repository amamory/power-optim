# Creates a DAG representing the dominance among all sequeces of frequencies for all islands.
# The starting node represents the max freq for each island and the subsequent nodes represent lower frequencies.
# The last node represents the lowest freq for each island.
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import sys

class Freq_DAG:
    # it assumes that the islands were previouly sorted by their capacity, i.e., 
    # num_freq_list[0] represents the frequency index of the island with the lowest performance
    def __init__(self, num_freq_list):
        self.root = list(num_freq_list)
        self.n_nodes = 0
        self.n_islands = len(self.root)
        self.edge_list = []
        self.node2freq = dict()
        self.node2freq_list = list()       
        self._create_dag()
        self._create_access_order()
    

    def _create_dag(self):
        # create the nodes and their attributed. this must be executed before creating the edges
        nodeData = self._create_node_idx()
        # how the nodes should be connected to represent the correct dominance effect
        self.edge_list = self._create_edges()
        # edge related data structures
        sources = [x[0] for x in self.edge_list]
        targets = [x[1] for x in self.edge_list]
        weights = [0 for x in range(len(self.edge_list))] # defined in runtime
        linkData = pd.DataFrame({'source' : sources,
                        'target' : targets,
                        'weight' :weights})
        self.G = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())
        nx.set_node_attributes(self.G, nodeData.set_index('name').to_dict('index'))


    # create the graph nodes
    # Nodes must be hashble types, so it's necessary to create integers to 
    # identify the nodes, while the list of frequency indexes is an atribute called 'freq'
    # The 'visited' attribute is used only during data navigation.
    def _create_node_idx(self):
        freqs = self._create_frequency_sequence()
        self.n_nodes = len(freqs)
        node_names = list(range(self.n_nodes))
        self.node2freq = dict(zip(node_names, freqs))
        self.node2freq_list = list(zip(node_names,freqs))
        false_list = [False]*self.n_nodes
        nodeData = pd.DataFrame(
                    {'name' : node_names,
                    'freq' : freqs,     # 
                    'visited' : false_list # 
                    })
        return nodeData
        #self.G.add_nodes_from(node_names)

    # creates a list with all combinations of frequency indexes
    def _create_frequency_sequence(self) -> list():
        freq_seq = []
        stop = False
        # start with all islands using their respective minimal frequencies
        freqs_per_island_idx = [0]*self.n_islands
        freq_seq.append(list(freqs_per_island_idx))
        while True:
            # points to the last incremented island
            inced = 0
            for i in range(self.n_islands):
                # if island idx i is not pointing to its max freq, then point to the next higher freq of this island
                if freqs_per_island_idx[i] < (self.root[i]):
                    freqs_per_island_idx[i] = freqs_per_island_idx[i] +1
                    inced = i
                    break
                else:
                    # if this is not the last island, then go to the next island to increment its freq
                    if i >= (self.n_islands-1):
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
        # TODO buggy! the 1st one is repeated. just deleating it for now
        del(reversed_freq_seq[0])
        return reversed_freq_seq

    def _get_freq_list(self,node_id) -> list():
        freq_list = [i[1] for i in self.node2freq_list if i[0] == node_id]
        # only one node is expected
        if len(freq_list) != 1:
            print ("ERROR: unexpected result in '_get_freq_list'")
            sys.exit(1)
        return freq_list[0]

    # return the node_id that matches the freq_list
    def _get_node(self,freq_list) -> int:
        if len(self.node2freq_list) == 0:
            print ('ERROR: the nodes must be initialized before using the graph')
            sys.exit(1)
        node_index = [i[0] for i in self.node2freq_list if i[1] == freq_list]
        # only one node is expected
        if len(node_index) == 0:
            print ('ERROR: no node found with freq list', freq_list)
            sys.exit(1)
        if len(node_index) > 1:
            print ('ERROR: multiple nodes found with freq list', freq_list)
            sys.exit(1)
        if not (node_index[0] >= 0 and node_index[0]<self.n_nodes):
            print ('ERROR: cannot find node with freq list', freq_list)
            sys.exit(1)
        return node_index[0]

    # return the edge list for the dag
    def _create_edges(self) -> list():
        # Stack to store all the nodes of the tree
        s1 = []
        edge_list = []
        # Push the root node
        s1.append(self.root)
        while len(s1) != 0:
            curr = s1.pop()
            source_node = self._get_node(curr)
            # for n islands, create up to n children nodes
            for i in range(len(curr)):
                if curr[i] == 0:
                    continue
                aux = list(curr)
                aux[i] = aux[i] -1
                s1.append(aux)
                target_node = self._get_node(aux)
                # saving the edge
                edge_list.append((source_node, target_node))
        return edge_list

    # for the same DAG, there might be multiple access approaches. 
    # this method goes from the max freq set to the lowest
    def _create_access_order(self):
        pass

    def plot(self):
        # pos = nx.spring_layout(self.G)
        # nx.draw(self.G, pos)
        # node_labels = nx.get_node_attributes(self.G,'freq')
        # nx.draw_networkx_labels(self.G, pos, labels = node_labels)
        # # edge_labels = nx.get_edge_attributes(self.G,'state')
        # # nx.draw_networkx_edge_labels(self.G, pos, labels = edge_labels)
        # plt.savefig('this.png')
        # plt.show()  

        # not using write_dot because the graph would not have the 'freq' node attribute
        #nx.drawing.nx_pydot.write_dot(self.G, 'graph.dot')
        f = open('graph.dot','w')
        f.write('strict digraph  {\n')
        for n in self.G.nodes:
            freq_list = self._get_freq_list(n)
            # create lines like this one
            # 0  [label="[2, 2, 2]"];
            line = "%d  [label=\"%s\"];\n" % (n,str(freq_list))
            f.write(line)
        for s,t in self.G.edges():
            line = "%d->%d;\n" % (s,t)
            f.write(line)
        f.write("}\n")
        f.close()




    # for debug purposes
    def print_dag():
        pass

    # go to the next frequency set, according to the created access order, skiping the sets 
    # already marked as not viable
    def next():
        pass

    # get the current frequency set
    def get() -> list():
        pass

    # set the current frequency set, and all its lower freq sets, as not viable solution
    def not_viable():
        pass


F = Freq_DAG([2,2,2])
F.plot()