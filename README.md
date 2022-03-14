# power-optim
Minizinc models of a power-aware task placement onto a heterogenous platform (.e.g, big-little, gpu, fpga).

## Setup 

```
$ sudo snap install minizinc
$ sudo apt-get install python3-venv
$ git clone https://github.com/amamory/power-optim.git
$ cd power-optim
$ python3 -m venv env
$ source env/bin/activate
$ pip install --editable .
$ pip install -r requirements.txt
```

## Models

Browse the directories to try each model. The suggested order in terms of complexity:
 - `makespan`: jobs without precedence constraint;
 - `makespan-dag`: similar to the previous one but jobs have precedence constraints represented by a DAG;
 - `deadline-dag`: almost equal to the previous one, but it includes an additional deadline constraint;
 - `deadline-power`: while the previous one minimizes the makespan, this one minizes the power as long as the deadline is respected;
 - `affinity`: this model is built on top of the previous one, but adding affinity constraint;
 - `pu-utilization`: this model is built on top of the previous one, but including *task period* and *processing unit utilization* constraint. **TODO**.

## Authors

 - Alexandre Amory (January 2022), [Real-Time Systems Laboratory (ReTiS Lab)](https://retis.santannapisa.it/), [Scuola Superiore Sant'Anna (SSSA)](https://www.santannapisa.it/), Pisa, Italy.

## Funding

This tool has been developed in the context of the [AMPERE project](https://ampere-euproject.eu). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871669.

