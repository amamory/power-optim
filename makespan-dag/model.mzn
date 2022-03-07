% The set of jobs (JOB) with  predence constraints modeled as a DAG,
% must be schedule on any of hw processing units (PU) represented by set CORE|.
% The jobs have a duration (d). 
% These hw PUs can be heterogeneous, meaning that for each job j in JOB, c1 and c2 in CORE, 
% the duration d[j,c1] can be different from d[j,c2]. Like, for example, in ARM bigLittle processors.

% Constraints:
% - a core can only run one job at a time
% - a job run once in a single core 
% - a job cannot be preempted
% - a job can only execute when all its preceeding jobs were executed

% Objective:
% - minimize the makespan (end)

%https://en.wikipedia.org/wiki/Open-shop_scheduling
%https://www.minizinc.org/doc-2.5.3/en/downloads/jobshop.mzn
%https://github.com/MiniZinc/minizinc-benchmarks/blob/master/openshop/openshop.mzn

% modeling graphs and dags
% http://www.hakank.org/minizinc/tsp.mzn
% https://github.com/MiniZinc/minizinc-benchmarks/blob/master/maximum-dag/maximum-dag.mzn
% https://github.com/MiniZinc/minizinc-benchmarks/blob/26bcd0a78433025f7b6896a6fa8caf128795760b/mapping/mesh2x2_1.dzn
% https://github.com/hakank/hakank/blob/master/minizinc/mvcp.mzn

% shortest path
% https://github.com/hakank/hakank/blob/master/minizinc/spp.mzn

% NoC model w CPU load
% https://github.com/MiniZinc/minizinc-benchmarks/blob/master/mapping/mapping.mzn

% MAIN DEFINITIONS
int : N_JOBS;
int : N_EDGES;
int : N_CORES;
enum JOB = anon_enum(N_JOBS);
enum EDGE = anon_enum(N_EDGES);
enum CORE = anon_enum(N_CORES);
array [JOB,CORE] of int: d;                   % task durations
array [1..N_EDGES,1..2] of 1..N_JOBS: e;            % the DAG edges: source node to target node

% BOUNDS FOR MAKESPAN 
% UPPER BOUND: since a job runs on a single workstation, lets be pessimistic and assume that the longest duration will be chosen for each pair job/core
% i.e., for each job, choose the longest duration
int: total = sum(i in JOB)(max(row(d,i)));  % total duration, upper bound for makespan
% LOWER BOUND: lets be optimistic and choose the shortest durations for each Job, sum their durations, and divide by the number of cores
int: lb = ceil(int2float(sum(i in JOB)(min(row(d,i)))) / int2float(card(CORE)));  % total duration, lower bound for makespan

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(total)));  % calculate the max number of digits just for printing the output
CORE: last = max(CORE);

% VARIABLES
array [JOB,CORE] of var 0..total: s;           % start times
array [JOB,CORE] of var 0..1: sel;             % 1 when the ws is selected to execute the job
var lb..total: end;                            % total end time, the variable representing the makespan
% aux variable used only to hold the selected initial time and duration for each job
% this will ease the DAG precedence constraint
% used[j,1]: initial time
% used[j,2]: duration
array [JOB,1..2] of var 0..total: used;

% a job runs on a single workstation
constraint forall(i in JOB)(
             sum(row(sel,i)) == 1);

constraint %% objective: minimize the makespan
    forall(j in CORE) (
        forall(i in JOB) (
            sel[i,j]*(s[i,j] + d[i,j]) <= end)
    );

% helper constraint to specify the DAG precedence
constraint
    forall(i in JOB)(
        forall(j in CORE where sel[i,j] == 1) (
            used[i,1] == s[i,j] /\ used[i,2] == d[i,j] 
        )
    );

% job0 has precedence over job1
predicate precedence(var int: start0, var int: duration0, var int: start1) =
  start0 + duration0 <= start1 ;

constraint % the DAG precedence
    forall(i in EDGE) (
        let { 
            var JOB: sj = to_enum(JOB,e[i,1]); % source job
            var JOB: tj = to_enum(JOB,e[i,2]); % target job
            } in 
            precedence (used[sj,1], used[sj,2], used[tj,1])
    );

predicate no_overlap(var int: start0, var int: duration0,
                     var int: start1, var int: duration1) =
  start0 + duration0 <= start1 \/ start1 + duration1 <= start0;
  
constraint %% ensure no overlap of tasks
    forall(j in CORE) (
        forall(i,k in JOB where i < k) (
            no_overlap(s[i,j],d[i,j], s[k,j],d[k,j])
        )
    );

solve minimize end;

output ["end = \(end)\n"] ++
       ["ub = \(total)\n"] ++
       ["lb = \(lb)\n"] ++
       ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in CORE 
       ] ++
       ["start_time, duration\n"] ++ 
       [ show_int(digs,used[i,j]) ++ " " ++ 
         if j == 2 then "\n" else "" endif |  
         i in JOB, j in 1..2 
       ];
