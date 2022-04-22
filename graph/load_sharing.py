# multiprocessing lib
# https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool
# https://stackoverflow.com/questions/4413821/multiprocessing-pool-example
# https://www.programcreek.com/python/example/3393/multiprocessing.Pool
# ray lib
# https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
# https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8
# https://docs.ray.io/en/latest/cluster/jobs-package-ref.html
import sys
import os
import subprocess
import multiprocessing as mp
import networkx as nx
import pandas as pd
import numpy as np
import math
import time
import yaml
import random

# libs
import tree

n_threads = 1
n_islands = 3
initial_task = 5
n = 1
while (n_threads > n_islands**n):
    n=n+1
end_task = initial_task-n+1
print (n, n_islands**n)
task_placements = tree.task_island_combinations(n_islands,initial_task,end_task)
for l in task_placements:
    print (l.data)
    for i in l.islands:
        print (i)
task_placements = [i.islands for i in task_placements]
task_placements = [[[6,5],[],[]]]
task_placements = [[[],[],[]]]

# Check whether it is possible to prune an entire branch of the search space.
# This might happen if the tasks placed so far are big enough to break the deadline or power constraints.
# If it's possible to prune, then just remove this item from the 'task_placements' list
def prune():
    pass

def search_best_placement(intial_placement) -> list():
    print (intial_placement, mp.current_process().name,mp.Process().name)
    search_tree = tree.Tree(intial_placement,initial_task,n_islands)
    n_leafs = 0
    placement = search_tree.get_next_leaf()
    while (placement):
        print (mp.current_process().name, placement)
        #time.sleep(random.randrange(2, 10))
        placement = search_tree.get_next_leaf()
        n_leafs += 1
    print (mp.current_process().name,'done!')
    return [n_leafs]


pool = mp.Pool(n_threads)
best_placement_list = []
for best_placement in pool.map(search_best_placement, task_placements):
    best_placement_list.append(best_placement)

print (best_placement_list)

pool.close()
pool.join()
