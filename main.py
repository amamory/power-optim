# Standard library imports
import sys
import os
import argparse
import time
import subprocess
import json
import pymzn
from operator import itemgetter # for sorting

# internal lib
#from common import plot_gantt

def main():
    """ Run Minizinc model and plot the resulting schedule 

    """

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Minizinc model.'
                        )
    parser.add_argument('data', type=str, 
                        help='Minizinc parameters file.'
                        )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print ("ERROR: File %s not found" % (args.model))
        exit(1)
    if not os.path.isfile(args.data):
        print ("ERROR: File %s not found" % (args.data))
        exit(1)

    # build a duration matrix out of the DZN file
    input_dict = pymzn.dzn2dict(str(args.data))
    #print(input_dict)
    n_jobs=input_dict["N_JOBS"]
    n_cores=input_dict["N_CORES"]
    d_list = []
    i=0
    for j in range(n_jobs):
        temp_list=[]
        for c in range(n_cores):
            temp_list.append(input_dict["d"][i])
            i = i+1
        d_list.append(temp_list)
    #print(d_list)

    path_model=args.model
    path_data=args.data

    out = None

    print ("MODEL:", path_model)
    print ("DATA:", path_data)
    try:
        out = subprocess.Popen(["minizinc", "-m", path_model, "-d", path_data,
            "--output-mode", "json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE )
    except subprocess.CalledProcessError as e:
        print(e.output)
        print ("ERROR: could not call minizinc")
        sys.exit(1)

    # CONVERT THE MINIZINC OUTPUT INTO JSON 
    
    # the minizinc output is a list of bytes
    out_str = out.stdout.readlines()
    # the 2 last lines are useless
    out_str = out_str[:-2]
    # this will convert a list of bytes into a single string
    out_json=b''.join(out_str).decode('utf-8')
    print("JSON output:")
    print(out_json)
    # convert json string into a dictionary, easing data access
    out_dict=json.loads(out_json)
    print(out_dict)
    print(out_dict["s"])

    # the plotting tool uses yaml format, so we need another convertion
    # yaml format: https://yatss.readthedocs.io/en/latest/#output-file-schedule-yaml-file
    sched = {}
    sched['title'] = 'Minizinc schedule'
    sched['sched'] = []
    for c in range(n_cores):
        sched_task = {}
        sched_task['name'] = "c"+str(c)
        sched_task['color'] = 'blue'
        sched_task['jobs'] = []
        for j in range(n_jobs):
            start_time=0
            end_time=0
            # if the job j has been assigned to core c, then get its start and end times
            if out_dict["sel"][j][c] == 1:
                start_time = out_dict["s"][j][c]
                end_time = start_time + d_list[j][c] - 1
                sched_task['jobs'].append([start_time,end_time])
        # sort the jobs in ascending order
        sched_task['jobs'] = sorted(sched_task['jobs'], key=itemgetter(0))
        sched['sched'].append(sched_task)    

    print(sched)
    #plot_gantt(sched)

if __name__ == "__main__":
    main()
