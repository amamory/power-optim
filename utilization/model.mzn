% The set of tasks (TASK) must be allocated to any of the hw processing units (PU) represented by set CORE|.
% The tasks have a WCET (C) and a minimum inter-arrival period (P). Their utilization is C/P
% The hw PUs can be heterogeneous, meaning that for each taski in JOB, c1 and c2 in CORE, 
% the WCET C[i,c1] can be different from C[i,c2]. Like, for example, in ARM big.LITTLE processors.

% Constraints:
% - a core can only host tasks up to saturation of its total utilization
% - a task can be allocated only to a single core 

% Objective:
% - minimize the maximum utilization among cores (max-spread policy, should be the most energy-savy one)

% MAIN DEFINITIONS
int : N_TASKS;
int : N_CORES;
enum TASK = anon_enum(N_TASKS);
enum CORE = anon_enum(N_CORES);
array [TASK,CORE] of int: C;                   % task worst-case durations
array [TASK] of int: P;                        % task min inter-arrival periods (equal to deadlines)

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(10000)));  % calculate the max number of digits just for printing the output
CORE: last = max(CORE);

% VARIABLES
array [TASK,CORE] of var 0..1: x;              % 1 when the core is selected to execute the task
var 0.0 .. 1.0 : max_util;                              % max utilization among cores

% a job runs on a single core
constraint forall(i in TASK)(
             sum(row(x,i)) == 1);

% a core hosts tasks up to its maximum utilization of 1.0
constraint forall(j in CORE)(
             sum(i in TASK)(int2float(x[i,j])*C[i,j]/int2float(P[i])) <= max_util);

solve minimize max_util;

output [ "Allocation matrix x[i,j]:\n" ] ++
       [ show_int(digs,x[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in TASK, j in CORE ] ++
       [ "------------\n"] ++
       [ "Core utilizations:\n" ] ++
       [ show_float(6,3,sum(i in TASK)(x[i,j]*C[i,j]/P[i])) ++ "\n" | j in CORE];
