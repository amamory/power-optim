
A model to minimize the power of the taskset, with precedence constraints modeled as a DAG, and map these tasks onto a heterogeneous set of cores. A deadline D is checked.

Note that `j0`, `c0` is chosen even considering that it is slower than `c1`. This is because `c0` is more energy efficient than `c1` and the deadline is still met.

```
$ source ./env/bin/activate
$ python main.py model.mzn simple.dzn
MODEL: model.mzn
DATA: simple.dzn
N_JOBS: 5
N_CORES: 2
JSON output:
{
  "s" : [[0, 12], [17, 11], [14, 10], [13, 2], [12, 0]],
  "sel" : [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]],
  "end" : 14,
  "total_energy" : 9,
  "used" : [[0, 10, 5], [11, 1, 1], [10, 1, 1], [13, 1, 1], [12, 1, 1]]
}

Makespan: 14
Total energy: 9
```

![](./simple.png "DAG for simple.dzn")
