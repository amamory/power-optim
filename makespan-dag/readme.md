
A model to minimize the makespan of the taskset, this time with precedence constraints modeled as a DAG, and map these tasks onto a heterogeneous set of cores. 

```
$ source ./env/bin/activate
$ python main.py model.mzn simple.dzn
MODEL: model.mzn
DATA: simple.dzn
N_JOBS: 5
N_CORES: 2
JSON output:
{
  "s" : [[0, 2], [1, 7], [8, 1], [7, 6], [6, 0]],
  "sel" : [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]],
  "end" : 7,
  "used" : [[0, 1], [1, 5], [1, 1], [6, 1], [6, 1]]
}

Makespan: 7
```

![](./simple.png "DAG for simple.dzn")

![](./sched.png "resulting schedule")
