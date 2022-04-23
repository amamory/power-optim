# multiprocessing lib
# https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool
# https://stackoverflow.com/questions/4413821/multiprocessing-pool-example
# https://www.programcreek.com/python/example/3393/multiprocessing.Pool
# distributed processes
# https://zditect.com/code/python/python-multiprocess-process-pool-data-sharing-process-communication-distributed-process.html
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



#GLOBALS

n_threads = 1
n_islands = 2
n_tasks = 5
# Check whether it is possible to prune an entire branch of the search space.
# This might happen if the tasks placed so far are big enough to break the deadline or power constraints.
# If it's possible to prune, then just remove this item from the 'task_placements' list
def prune():
    pass

# To pass the shared variables to the pool.
# like shared values, locks cannot be shared in a Pool - instead, pass the 
# multiprocessing.Lock() at Pool creation time, using the initializer=init_lock.
# This will make your lock instance global in all the child workers.
# The init_globals is defined as a function - see init_globals() at the top.
# source : https://serveanswer.com/questions/multiprocessing-pool-map-multiple-arguments-with-shared-value-resolved
# source: https://stackoverflow.com/questions/53617425/sharing-a-counter-with-multiprocessing-pool
def init_globals(counter, l):
    global shared_best_power
    global shared_lock
    shared_best_power = counter
    shared_lock = l

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
    print ('\nStarting work load:\n', ' - initial placement:', placement_setup[0], '\n  - # placements:', placement_setup[1], '\n  - process:', mp.current_process().name,mp.Process().name)
    current_placement = placement_setup[0]
    n_placements = placement_setup[1]

    # testing the lock mechanism
    with shared_lock:
        best_power = shared_best_power.value
        shared_best_power.value = best_power + 1 
        print (shared_best_power.value)

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

def main():

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


    # shared variable used to keep the the processes updated related to the lowest power consumption found so far.
    # this is used to bound the search space
    manager = mp.Manager()
    # 'd' means double precision, 'i' is integer
    shared_best_power = manager.Value('d', 999999.0)
    shared_lock = mp.Lock()


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

    # Alternatively, you could use Pool.imap_unordered, which starts returning 
    # results as soon as they are available instead of waiting until everything is finished. So you could tally the amount of returned results and use that to update the progress bar.
    # source: https://devdreamz.com/question/633149-how-to-use-values-in-a-multiprocessing-pool-with-python
    best_placement_list = []
    pool =  mp.Pool(initializer=init_globals, processes=n_threads, initargs=(shared_best_power,shared_lock,))
    result_list = pool.map(search_best_placement, placement_setup_list)
    for best_placement in result_list:
        best_placement_list.append(best_placement)

    pool.close()
    pool.join()

    # if the stating value is 
    print ('final power:', shared_best_power.value)

