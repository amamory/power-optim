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
import multiprocessing as mp
import networkx as nx
import pandas as pd
import numpy as np
import math
import time

n_threads = 3
n_islands = 2
n_tasks = 5
n = 1
while (n_threads > n_islands**n):
    n=n+1

# the size of the entire search space of 'n_tasks' placements onto 'n_islands'
total_task_placements = n_islands**n_tasks
# it represents how many work packages i can divide the entire search space (total_task_placements)
# of placements considering a parallelism of 'n_threads'
search_space_subdivisions = n_islands**n
# the size of each work package, i.e., the number of placements to be checked by work package
placements_per_workload = int (math.ceil(float(total_task_placements) / float(search_space_subdivisions)))
print ('total_task_placements:',total_task_placements)
print ('search_space_subdivisions:', search_space_subdivisions)
print ('placements_per_workload:', placements_per_workload)

# Generate the data structure sent to the pool of 'threads', i.e., the initial placement and 
# how many placements each work package must check. 
# The way it's calculated, all work package should have the same size, regardeless 
# n_threads, n_islands, n_tasks
placement_cnt = 0
placement_setup_list = []
while (placement_cnt < total_task_placements):
    # initial placement id and the number of placements to be generated from this initial one
    placement_setup_list.append((placement_cnt,placements_per_workload))
    placement_cnt += placements_per_workload
    
print ('work packages:', placement_setup_list)

# Check whether it is possible to prune an entire branch of the search space.
# This might happen if the tasks placed so far are big enough to break the deadline or power constraints.
# If it's possible to prune, then just remove this item from the 'task_placements' list
def prune():
    pass

# converts an integer in to a task mapping, i.e., a list of list
def get_mapping(curr_mapping, n_islands, n_tasks) -> list():
    mapping = [[] for i in range(n_islands)]
    island = 0
    # compute island onto which I'm mapping the i-th task
    for i in range(n_tasks): # scanning through all the n tasks
        island = int(curr_mapping / (n_islands**(i)) % n_islands)
        mapping[island].append(i)
    return mapping

# function to test whether the Pool mechanism and task mapping sequencer are generating 
# all the possible task mappings
def search_best_placement(placement_setup) -> list():
    print ('starting work load:', placement_setup, mp.current_process().name,mp.Process().name)
    current_placement = placement_setup[0]
    n_placements = placement_setup[1]
    # f = open(mp.Process().name+'.txt','w+')
    # f.write(str(placement_setup)+'\n')
    for i in range(n_placements):
        placement = get_mapping(current_placement, n_islands, n_tasks)
        print (mp.current_process().name, placement)
        # f.write(str(placement)+'\n')
        current_placement += 1
    print (mp.current_process().name,'done!')
    # f.close()
    return []


pool = mp.Pool(n_threads)
best_placement_list = []
for best_placement in pool.map(search_best_placement, placement_setup_list):
    best_placement_list.append(best_placement)

print (best_placement_list)

pool.close()
pool.join()
