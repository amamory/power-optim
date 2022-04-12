# Creates a DAG representing the dominance among all sequeces of frequencies for all islands.
# The starting node represents the max freq for each island and the subsequent nodes represent lower frequencies.
# The last node represents the lowest freq for each island.
import networkx as nx
import pandas as pd
import sys

class Freq_DAG:
    # it assumes that the islands were previouly sorted by their capacity, i.e., 
    # num_freq_list represents the number of frequencies on each island.
    # num_freq_list[0] represents the frequency index of the island with the lowest performance
    def __init__(self, num_freq_list):
        # the root is a index to the freq set with highest frequency for each island
        self.root = [i-1 for i in num_freq_list]
        self.n_nodes = 0
        self.n_islands = len(self.root)
        self.node2freq_list = list()
        # the dag
        self.G = self._create_dag()
        # this indexes are used when transversing the graph in the deacresing freq order
        self.access_order = self._create_access_order()
        self.curr_node_idx = 0
        # performance counters to check how many times each freq is checked
        # freq_cnts[i][0] for viable solutions
        # freq_cnts[i][1] for NOT viable solutions
        self.freq_cnts = [(0,0) for i in range(len(self.node2freq_list))]
    
    # it creates the dag that represents the dominance among the freq sequences
    def _create_dag(self):
        # create the nodes and their attributed. this must be executed before creating the edges
        nodeData = self._create_node_idx()
        # how the nodes should be connected to represent the correct dominance effect
        edge_list = self._create_edges()
        # edge related data structures
        sources = [x[0] for x in edge_list]
        targets = [x[1] for x in edge_list]
        weights = [0] * len(edge_list)
        linkData = pd.DataFrame({'source' : sources,
                    'target' : targets,
                    'weight' :weights})
        dag = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())
        nx.set_node_attributes(dag, nodeData.set_index('name').to_dict('index'))
        return dag


    # create the graph nodes
    # Nodes must be hashble types, so it's necessary to create integers to 
    # identify the nodes, while the list of frequency indexes is an atribute called 'freq'
    # The 'skip' attribute is used only during data navigation.
    def _create_node_idx(self):
        freqs = self._create_frequency_sequence()
        self.n_nodes = len(freqs)
        node_names = list(range(self.n_nodes))
        # list used to map node_id with an unique freq seq
        # TODO, in the future, node_names could be dropped because it conincides with the list index
        self.node2freq_list = list(zip(node_names,freqs))
        false_list = [False]*self.n_nodes
        nodeData = pd.DataFrame(
                    {'name' : node_names,
                    'freq' : freqs,     # 
                    'skip' : false_list # 
                    })
        return nodeData

    # creates a list with all combinations of frequency indexes
    # the order they are generated is not relevant for this application
    # the important is that it must have a one-to-one relationship among 
    # node id and a freq sequence
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
        # Stack to store all the nodes of the dag
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
    # this method goes from the max freq set to the lowest. 
    # It initially seems very similar to '_create_edges' but the
    # access pattern is different. For instance, in this function the item 
    # taken from the stack is the first one, not the last.
    def _create_access_order(self):
        # Stack to store all the nodes of the dag
        s1 = []
        access_order = []
        # Push the root node
        s1.append(self.root)
        while len(s1) != 0:
            curr = s1.pop(0)
            source_node = self._get_node(curr)
            if source_node not in access_order:
                access_order.append(source_node)
            # for n islands, create up to n children nodes
            for i in range(len(curr)):
                if curr[i] == 0:
                    continue
                aux = list(curr)
                aux[i] = aux[i] -1
                s1.append(aux)
        # check if the order is the expected on
        # idx=0
        # for i in access_order:
        #     print (idx,self.G.nodes[i]['freq'])
        #     idx = idx +1
        return access_order

    def print_dag(self):
        for n in range(self.n_nodes):
            print (n, self.G.nodes[n]['skip'],self.G.nodes[n]['freq'],)

    # Generates a .dot file to be used with graphviz for debuging
    def plot(self):
        # 1) This option results into a very bad layout. Other nx layout didnt help that much
        # pos = nx.spring_layout(self.G)
        # nx.draw(self.G, pos)
        # node_labels = nx.get_node_attributes(self.G,'freq')
        # nx.draw_networkx_labels(self.G, pos, labels = node_labels)
        # # edge_labels = nx.get_edge_attributes(self.G,'state')
        # # nx.draw_networkx_edge_labels(self.G, pos, labels = edge_labels)
        # plt.savefig('this.png')
        # plt.show()  

        # 2) not using write_dot because the graph would not have the 'freq' node attribute
        #nx.drawing.nx_pydot.write_dot(self.G, 'graph.dot')

        # 3) The best approach was to manually write the dot file
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

    # used to start over another search whihout creating the whole dag again
    def reinitiate_dag(self):
        self.curr_node_idx = 0
        for n in self.G.nodes():
            self.G.nodes[n]['skip'] = False

    # get the performance counters
    def get_counters(self):
        return self.freq_cnts

    # go to the next frequency set, according to the created access order, skiping the sets 
    # already marked to be skipped
    def next(self):
        if self.curr_node_idx+1 > self.n_nodes:
            self.curr_node_idx = 0
            # I suppose it is not expected to get to this point
            print ('ERROR: not expected to reach this point in function "next"')
            sys.exit(1)
        # when next is called, it means that, if this node was not marked to be skiped, then this is a viable solution 
        curr_node = self.access_order[self.curr_node_idx]
        if not self.G.nodes[curr_node]['skip']:
            aux_tuple = (self.freq_cnts[curr_node][0] + 1, self.freq_cnts[curr_node][1])
            self.freq_cnts[curr_node] = aux_tuple

        found = False
        initial_idx = self.curr_node_idx
        # keep the loop until a node that should not be skipped if found or reach the end of the dag
        for n in self.access_order[self.curr_node_idx+1:]:
            initial_idx = initial_idx +1
            if not self.G.nodes[n]['skip']:
                found = True
                break
        # stop the index in the last valid node
        if found:
            self.curr_node_idx = initial_idx
        return found

    # get the current frequency set
    def get(self) -> list():
        curr_node = self.access_order[self.curr_node_idx]
        return self._get_freq_list(curr_node)

    # set the current frequency set, and all its lower freq sets, as not viable solution
    # by marking their 'skip' attribute
    def not_viable(self):
        # mark all nodes from the current node until the last one with 'skip' = True
        curr_node = self.access_order[self.curr_node_idx]
        # the curr node was said to be not a viable solution
        aux_tuple = (self.freq_cnts[curr_node][0], self.freq_cnts[curr_node][1] + 1)
        self.freq_cnts[curr_node] = aux_tuple
        # Stack to store all the nodes of the dag
        s1 = []
        # Push the current node
        s1.append(curr_node)
        while len(s1) != 0:
            curr_node = s1.pop(0)
            self.G.nodes[curr_node]['skip'] = True
            # push in to the stack all the outgoing nodes from the current node
            for s,t in self.G.out_edges(curr_node):
                s1.append(t)

# Testing
# F = Freq_DAG([2,2,2])
# F._create_access_order()
# F.reinitiate_dag()
# print (F.get())
# F.next()
# print (F.get())
# F.next()
# print (F.get())
# F.next()
# print (F.get())
# F.not_viable()
# F.print_dag()
# F.next()
# print (F.get())
# F.next()
# print (F.get())
# F.next()
# print (F.get())
# F.next()
# print (F.get())
# print (F.next())
# print (F.get())
# print (F.next())
# print (F.get())
# print (F.next())
# print (F.get())
