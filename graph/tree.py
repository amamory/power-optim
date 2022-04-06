# Generate all combinations of t task placement in i islands
# inital source:
# https://www.geeksforgeeks.org/print-all-leaf-nodes-of-a-binary-tree-from-left-to-right-set-2-iterative-approach/?ref=lbp

# n-ary tree node
class Node:
    def __init__(self, data, parent, nary=2):
        self.data = data
        self.parent = parent # not in use at this moment
        # store the tasks in each island
        self.islands = []
        for i in range(nary):
            self.islands.append([])
        self.children = [None]*nary

# number of nodes of the tree
node_id = 0
# the resulting list of leafs
list_of_leafs = []

# function that returns all combinations of tasks placement onto islands
def task_island_combinations(n_islands, n_tasks) -> list():
    global list_of_leafs

    # n_islands = 3
    # n_tasks = 3
    n_leafs = num_leafs_perfect_tree(n_islands,n_tasks)
    print('Islands:',n_islands, 'tasks:', n_tasks, 'combinations:', n_leafs)
    root = _build_perfect_tree(n_islands,n_tasks+1, None,None)
    print ('n_nodes',node_id)
    _generate_leafs(root)

    return list_of_leafs

# calculate the number of leafs of a perfect tree of n-ary and height h
def num_leafs_perfect_tree(ary,h) -> int:
    return int (((ary**(h+1)-1)/(ary-1)) - ((ary**(h)-1)/(ary-1)))

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
                #break
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
def _build_perfect_tree(ary,height, branch, islands) -> Node:
    global node_id
    node = Node(node_id, None, ary)
    node_id = node_id + 1
    if branch is not None:
        # only the root must be branch==None
        # copy the island fro the parent and add another task
        node.islands = [x[:] for x in islands]
        node.islands[branch].append(height-1)
    if height == 1:
        return node
    for i in range(ary):
        node.children[i] = _build_perfect_tree(ary, height-1, i, node.islands)
    return node

if __name__ == "__main__":

    leaf_list = task_island_combinations(3,3)
    # check the list of leafs
    for l in leaf_list:
        print (l.data)
        for i in l.islands:
            print (i)
