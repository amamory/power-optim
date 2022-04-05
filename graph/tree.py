
# source
# https://www.geeksforgeeks.org/print-all-leaf-nodes-of-a-binary-tree-from-left-to-right-set-2-iterative-approach/?ref=lbp

# Python3 program to print all the leaf
# nodes of a Binary tree from left to right

# Binary tree node
class Node:

    def __init__(self, data, parent, nary=2):
        self.data = data
        self.parent = parent
        self.children = [None]*nary

# Function to Print all the leaf nodes
# of Binary tree using two stacks
def PrintLeafLeftToRight(root):

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

        # # If current node has a left child
        # # push it onto the first stack
        # if curr.left:
        #     s1.append(curr.left)

        # # If current node has a right child
        # # push it onto the first stack
        # if curr.right:
        #     s1.append(curr.right)
        

        # # If current node is a leaf node 
        # # push it onto the second stack
        # elif not curr.left and not curr.right:
        #     s2.append(curr)

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
        print(s2.pop().data, end = " ")
	
node_id = 0
# https://www.programiz.com/dsa/perfect-binary-tree
def build_perfect_tree(ary,height) -> Node:
    global node_id
    node = Node(node_id, None, ary)
    node_id = node_id + 1
    if height == 1:
        return node
    for i in range(ary):
        node.children[i] = build_perfect_tree(ary, height-1)
    return node

# Driver code
if __name__ == "__main__":

    # root = Node(1,None)
    # root.children[0] = Node(2,root)
    # root.children[1] = Node(3, root)
    # root.children[0].children[0] = Node(4, root.children[0])
    # root.children[1].children[0] = Node(5, root.children[1])
    # root.children[1].children[1] = Node(7, root.children[1])
    # root.children[0].children[0].children[0] = Node(10, root.children[0].children[0])
    # root.children[0].children[0].children[1] = Node(11, root.children[0].children[0])
    # root.children[1].children[1].children[0] = Node(8,root.children[1].children[1])
    root = build_perfect_tree(2,3)

    print ('n_nodes',node_id)
    PrintLeafLeftToRight(root)

# This code is contributed
# by Rituraj Jain
