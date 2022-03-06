# reused from https://github.com/amamory-embedded/sched-learning/blob/master/src/common.py
import sys
from math import gcd
import datetime
# for plotting
import plotly.express as px
import pandas as pd
import numpy as np


def versiontuple(v):
    """Convert a string of package version in a tuple for future comparison.

    :param v: string version, e.g "2.3.1".
    :type  v: str
    :return: The return tuple, e.g. (2,3,1).
    :rtype:  tuple 

    :Example: 

        >>> versiontuple("2.3.1") > versiontuple("10.1.1")
        >>> False
    """    
    return tuple(map(int, (v.split("."))))



def check_sched(sched):
    """Parse the YAML for the resulting schedule of a scheduling algorithm.

    .. literalinclude:: ../../wikipedia-sched.yaml
        :language: yaml
        :linenos:

    :param task_list: List of sched descriptors, as in the example above.
    :type  task_list: List of dictionaries.
    :return: True for success, False otherwise.
    :rtype: bool
    """
    
    ##############
    # validate the input format first. some fields are expected for rms
    ##############

    # must have at least 1 task
    if len(sched['sched']) < 1:
        print ("ERROR: the sched list must have at least 1 task. Found", len(sched['sched']))
        return False

    # check if all tasks have the mandatory fields
    print ('checking the scheduling list ... ', end='')
    for task in sched['sched']:
        if 'name' not in task:
            print ("\nERROR: field 'name' not found in task")
            return False
        if 'jobs' not in task:
            print ("\nERROR: field 'jobs' not found in task")
            return False
        if len(task['jobs']) <= 0:
            print ("\nWARNING: task %s has no job. Got" % task['name'], len(task['jobs']))
            #return False
        for job in task['jobs']:
            if type(job[0]) is not int or type(job[1]) is not int:
                print ("\nERROR: jobs must be int initial and final times. Got", type(job[0]), type(job[1]))
                return False
            if job[0] > job[1]:
                print ("\nERROR: the initial job time must be lower than the the final time. Got", job[0], job[1])
                return False
            # zero is not supported in the plotting function
            if job[0] < 0:
                print ("\nERROR: the initial job time must be greater than 0. Got", job[0])
                return False
            if job[1] < 0:
                print ("\nERROR: the initial job time must be greater than 0. Got", job[1])
                return False

    print ('passed !')  
    return True  

def convert_to_datetime(x):
    """Converts a natural number to date. 
    
    It converts a natural number to date, where 0 corresponds to 
    datetime(1970, 1, 1) (assuming, y/m/d). Value 1 corresponds to 
    datetime(1970, 1, 2).

    :param x: natural value
    :type  x: int

    :rtype: datetime
    """

    #data_conv = datetime.datetime.fromtimestamp(31536000+x*24*3600).strftime("%Y-%m-%d")
    data_conv2 = (datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=x)).strftime("%Y-%m-%d")
    return data_conv2

def plot_gantt(sched, verbose = False):
    """Use the plotly lib to plot the gantt chart.


    .. literalinclude:: ../../wikipedia-sched.yaml
        :language: yaml
        :linenos:

    :param  sched: The shedule YAML file, as in the example above.
    :type   sched: List of dictionaries.
    :param  verbose: enable/disable verbose mode
    :type   verbose: bool
    :return: None

    .. todo:: add a slider

        https://plotly.com/python/animations/

        https://plotly.com/python/sliders/

    .. todo:: provide an alternative plotting option with matplotlib

        https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/broken_barh.html

    .. todo:: other alternative plots

        https://github.com/ehsan-elwan/RM-Task-Scheduling/blob/master/Plotter.py

        https://github.com/johnharakas/scheduling-des/blob/sim-plotting/Qt_Canvas.py

        https://github.com/esalehi1996/Realtime_Scheduling_python/blob/master/main.py

        https://github.com/carlosgeos/uniprocessor-scheduler/blob/master/src/simulation.py

        https://github.com/ksameersrk/rt-scheduler/blob/master/analysis/plot_graph.py

        https://github.com/guilyx/gantt-trampoline/blob/master/lib/GanttPlot.py

    .. todo:: export figure with the plot
        https://plotly.com/python/static-image-export/        
    """

    # check plotly version
    import plotly as pl
    if versiontuple(pl.__version__) < versiontuple("4.9.0"):
        print ("ERROR: the scheduling plotting function requires plotly 4.9.0 or newer. Found", pl.__version__)
        return False

    # check the input argument
    if not check_sched(sched):
        print("Aborting execution of scheduling plotting due to invalid input file.")
        sys.exit(1)

    # get the max value of x of the schedule to be used in the plot
    max_x = 0
    # create the data format required by pandas DataFrame
    list_tasks = []
    for task in sched['sched']:
        # color is not mandatory for a task
        if 'color' in  task:
            task_color = task['color']
        else:
            task_color = 'blue' # the default color
        if len(task['jobs']) == 0:
            # place the task in the char even if it had no job executed
            list_tasks.append(dict(Task=task['name'], Start=convert_to_datetime(0), 
                    Finish=convert_to_datetime(0), Color = task_color, 
                    # used only by the hover feature
                    Start_tick = 0, Finish_tick = 0, Duration = 0
                    ))
        else:
            for job in task['jobs']:
                list_tasks.append(dict(Task=task['name'], Start=convert_to_datetime(job[0]), 
                    Finish=convert_to_datetime(job[1]), Color = task_color, 
                    # used only by the hover feature
                    Start_tick = job[0], Finish_tick = job[1], Duration = job[1]-job[0]
                    ))
                max_x = max(max_x,max(job[0],job[1]))

    # creating the pandas DataFrame requred by plotly
    df = pd.DataFrame(list_tasks)

    # title is optional
    if 'title' in  sched:
        chart_title = sched['title']
    else:
        chart_title = ''

    fig = px.timeline(df, title = chart_title, x_start="Start", x_end="Finish", color = "Color", y="Task",
        # info used only for the hover feature
        custom_data = ['Start_tick','Finish_tick','Duration'])
    
    fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up

    # format the data appearing in the mouse hover feature
    fig.update_traces(
        hovertemplate = "Start:%{customdata[0]}<br>End: %{customdata[1]}<br>Duration: %{customdata[2]}"
    )
    
    # this part converts dates into ticks
    num_tick_labels = np.linspace(start = 0, stop = max_x, num = max_x+1, dtype = int)
    date_ticks = [convert_to_datetime(int(x)) for x in num_tick_labels]
    fig.layout.xaxis.update({
            'tickvals' : date_ticks,
            'ticktext' : num_tick_labels
            })

    # it will show in the browser
    fig.show()
