% The set of jobs (JOB) must be schedule on any of hw processing units (PU) represented by set CORE|.
% The jobs have a duration (d). 
% These hw PUs can be heterogeneous, meaning that for each job j in JOB, c1 and c2 in CORE, 
% the duration d[j,c1] can be different from d[j,c2]. Like, for example, in ARM bigLittle processors.

% Constraints:
% - a core can only run one job at a time
% - a job run once in a single core 
% - a job cannot be preempted

% Objective:
% - minimize the makespan (end)

%https://en.wikipedia.org/wiki/Open-shop_scheduling
%https://www.minizinc.org/doc-2.5.3/en/downloads/jobshop.mzn
%https://github.com/MiniZinc/minizinc-benchmarks/blob/master/openshop/openshop.mzn
     
% MAIN DEFINITIONS
int N_JOBS;
int N_CORES;
enum JOB = anon_enum(N_JOBS);
enum CORE = anon_enum(N_CORES);
array [JOB,CORE] of int: d;                   % task durations

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

% a job runs on a single workstation
constraint forall(i in JOB)(
             sum(row(sel,i)) == 1);

constraint %% minimize the makespan
    forall(j in CORE) (
        forall(i in JOB) (
            sel[i,j]*(s[i,j] + d[i,j]) <= end)
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
       ["------------\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in CORE ] ++
       ["------------\n"] ++ 
       [
           ( if fix(sel[i,j]) = 1 then show_int(digs,j) ++ ": " ++ show_int(digs,s[i,j]) ++ " " ++ show_int(digs,d[i,j]) ++ "\n" else "" endif ) 
           |i in JOB, j in CORE
       ] ;
         