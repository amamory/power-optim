# Generate all combinations of t task placement in i islands
# inital source:
# https://www.geeksforgeeks.org/print-all-leaf-nodes-of-a-binary-tree-from-left-to-right-set-2-iterative-approach/?ref=lbp


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


# n-ary tree node
class Node:
    def __init__(self, data, parent, nary=2):
        self.data = data
        self.parent = parent # not in use at this moment
        # store the tasks in each island
        self.islands = [[] for i in range(nary)]
        self.children = [None]*nary

class Tree:
    def __init__(self, intial_placement, height, nary=2):
        self.root = Node(0,None,nary)
        self.root.islands = list(intial_placement[:])
        self.current_leaf = None
        self.tree_height = height - sum([len(i) for i in intial_placement])
        self.nary = nary
        self.node_id = 0
        # grow the tree
        self._grow()
        self._get_first_leaf()
    
    def _grow(self):
        self._generate_node(self.nary, self.tree_height, 0, self.root)

    def _get_first_leaf(self):
        node = self.root
        # reach the left hand side of the tree
        while (node.children[0] != None):
            node = node.children[0]
        self.current_leaf = node

    def get_next_leaf(self,node) -> None:
        leaf =  self.current_leaf
        # if all tasks are in the right hand side
        if len(leaf.children[-1]) == self.tree_height:
            return None
        else:
            self.current_leaf = self.current_leaf.parent
            return self.current_leaf

    # def get_current_leaf(self):
    #     return self.current_leaf

    def _generate_node(self, ary, height, branch, parent) -> Node:
        self.node_id += 1
        node = Node(self.node_id, parent, ary)
        if branch is not None:
            # only the root must be branch==None
            # copy the island fro the parent and add another task
            node.islands = [x[:] for x in parent.islands]
            node.islands[branch].append(height-1)
        if height == 1:
            print (node.data, node.islands)
            return node
        for i in range(ary):
            node.children[i] = self._generate_node(ary, height-1, i, node)
        return node

# number of nodes of the tree
node_id = 0
# the resulting list of leafs
list_of_leafs = []

# function that returns all combinations of tasks placement onto islands
def task_island_combinations(n_islands, n_tasks, end_height=1) -> list():
    global list_of_leafs

    # n_leafs = num_leafs_perfect_tree(n_islands,n_tasks)
    n_leafs = n_islands**(n_tasks+1-end_height)
    print('Islands:',n_islands, 'tasks:', n_tasks, 'combinations:', n_leafs)
    root = _build_perfect_tree(n_islands,n_tasks+1, end_height, None,None)
    print ('n_nodes',node_id)
    _generate_leafs(root)

    return list_of_leafs

# calculate the number of leafs of a perfect tree of n-ary and height h
# this is equivalent to ary**h
#def num_leafs_perfect_tree(ary,h) -> int:
    #return int (((ary**(h+1)-1)/(ary-1)) - ((ary**(h)-1)/(ary-1)))

# generate the leafs of a perfect tree of n-ary and height h
def _generate_leafs(root):
    global list_of_leafs

    # Stack to store all the nodes
    # of tree
    s1 = []

    # Stack to store all the
    # leaf nodes
    s2 = []

    # Push the root node
    s1.append(root)

    while len(s1) != 0:
        curr = s1.pop()
        # If current node has a child, 
        # push it onto the first stack
        empty = True
        for i in curr.children:
            if i:
                s1.append(i)
                empty = False
        # If current node is a leaf node 
        # push it onto the second stack
        if empty:
            s2.append(curr)

    # Print all the leaf nodes
    while len(s2) != 0:
        # print(s2.pop().data, end = " ")
        # save the leaf in the list
        list_of_leafs.append(s2.pop())
	
# build the perfect tree
# initial source: https://www.programiz.com/dsa/perfect-binary-tree
def _build_perfect_tree(ary,height, end_height, branch, islands) -> Node:
    global node_id
    node = Node(node_id, None, ary)
    node_id = node_id + 1
    if branch is not None:
        # only the root must be branch==None
        # copy the island fro the parent and add another task
        node.islands = [x[:] for x in islands]
        node.islands[branch].append(height-1)
    if height == end_height:
        return node
    for i in range(ary):
        node.children[i] = _build_perfect_tree(ary, height-1, end_height, i, node.islands)
    return node

if __name__ == "__main__":

    # task_island_combinations(3,7,6)
    # generates the placement sequence of tasks 6 and 5, using a tree of height 2 == 7+1-6
    # task_island_combinations(3,2)
    # generates the placement sequence of tasks 1 and 0, using a tree of height 2 == 2+1-1
    leaf_list = task_island_combinations(3,7,6)
    # check the list of leafs
    for l in leaf_list:
        print (l.data)
        for i in l.islands:
            print (i)
