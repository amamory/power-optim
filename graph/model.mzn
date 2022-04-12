% The set of tasks (TASK) with  predence constraints modeled as a DAG,
% must be schedule on any of hw processing units (PU) represented by set CORE|.
% The tasks have a duration (T). 

% Constraints:
% - a core can only run one job at a time
% - a job run once in a single core 
% - a job cannot be preempted
% - a job can only execute when all its preceeding tasks were executed

% Objective:
% - minimize the makespan (end)

% MAIN DEFINITIONS
int : N_TASKS;
int : N_EDGES;
int : N_CORES;
int : Ddag;
enum TASK = anon_enum(N_TASKS);
enum EDGE = anon_enum(N_EDGES);
enum CORE = anon_enum(N_CORES);
array [TASK] of int: D;                   % task deadline
array [TASK] of int: T;                   % task wcet
array [1..N_EDGES,1..2] of 1..N_TASKS: E;      % the DAG edges: source node to target node

% BOUNDS FOR MAKESPAN 
% UPPER BOUND: since a job runs on a single workstation, lets be pessimistic and assume that the longest duration will be chosen for each pair job/core
% i.e., for each job, choose the longest duration
int: total = sum(i in TASK)(T[i]);  % total duration, upper bound for makespan
% LOWER BOUND: lets be optimistic and choose the shortest durations for each Job, sum their durations, and divide by the number of cores
%int: lb = ceil(int2float(sum(T))) / int2float(card(CORE));  % total duration, lower bound for makespan
int: lb = 0;

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(total)));  % calculate the max number of digits just for printing the output
CORE: last = max(CORE);

% VARIABLES
array [TASK,CORE] of var 0..total: s;           % start times
array [TASK,CORE] of var 0..1: sel;             % 1 when the ws is selected to execute the job
var lb..total: end;                            % total end time, the variable representing the makespan
% aux variable used only to hold the selected initial time and duration for each job
% this will ease the DAG precedence constraint
% used[j,1]: initial time
% used[j,2]: duration
array [TASK,1..2] of var 0..total: used;

% a job runs on a single workstation
constraint forall(i in TASK)(
             sum(row(sel,i)) == 1);

constraint %% objective: minimize the makespan
    forall(j in CORE) (
        forall(i in TASK) (
            sel[i,j]*(s[i,j] + T[i]) <= end)
    );

constraint end <= Ddag;

% helper constraint to specify the DAG precedence
constraint
    forall(i in TASK)(
        forall(j in CORE where sel[i,j] == 1) (
            used[i,1] == s[i,j] /\ used[i,2] == T[i] 
        )
    );

% job0 has precedence over job1
predicate precedence(var int: start0, var int: duration0, var int: start1) =
  start0 + duration0 <= start1 ;

constraint % the DAG precedence
    forall(i in EDGE) (
        let { 
            var TASK: sj = to_enum(TASK,E[i,1]); % source job
            var TASK: tj = to_enum(TASK,E[i,2]); % target job
            } in 
            precedence (used[sj,1], used[sj,2], used[tj,1])
    );

predicate no_overlap(var int: start0, var int: duration0,
                     var int: start1, var int: duration1) =
  start0 + duration0 <= start1 \/ start1 + duration1 <= start0;
  
constraint %% ensure no overlap of tasks
    forall(j in CORE) (
        forall(i,k in TASK where i < k) (
            no_overlap(s[i,j],T[i], s[k,j],T[k])
        )
    );

% solve minimize end;
solve satisfy;

output ["end = \(end)\n"] ++
       ["ub = \(total)\n"] ++
       ["lb = \(lb)\n"] ++
       ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in TASK, j in CORE 
       ] ++
       ["start_time, duration\n"] ++ 
       [ show_int(digs,used[i,j]) ++ " " ++ 
         if j == 2 then "\n" else "" endif |  
         i in TASK, j in 1..2 
       ];
