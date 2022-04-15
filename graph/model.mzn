% The set of nodes (NODE) with  predence constraints modeled as a DAG,
% (note: except for the first and the last nodes, the other nodes represents tasks)
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
int : N_NODES;
int : N_EDGES;
int : N_CORES;
int : N_UNREL_ROWS;
int : N_UNREL_COLS;
int : Ddag;
enum NODE = anon_enum(N_NODES);
enum EDGE = anon_enum(N_EDGES);
enum CORE = anon_enum(N_CORES);
array [NODE] of int: D;                   % task deadline
array [NODE] of int: T;                   % task wcet
array [1..N_NODES] of float: U;                 % CPU utilization per task
array [1..N_EDGES,1..2] of 1..N_NODES: E;      % the DAG edges: source node to target node

array [1..N_UNREL_ROWS,1..N_UNREL_COLS] of 1..N_NODES: unrelated_node;

% BOUNDS FOR MAKESPAN 
% UPPER BOUND: since a job runs on a single workstation, lets be pessimistic and assume that the longest duration will be chosen for each pair job/core
% i.e., for each job, choose the longest duration
int: total = sum(i in NODE)(T[i]);  % total duration, upper bound for makespan
% LOWER BOUND: lets be optimistic and choose the shortest durations for each Job, sum their durations, and divide by the number of cores
%int: lb = ceil(int2float(sum(T))) / int2float(card(CORE));  % total duration, lower bound for makespan
int: lb = 0;

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(total)));  % calculate the max number of digits just for printing the output
CORE: last = max(CORE);

% VARIABLES
array [NODE,CORE] of var 0..total: s;           % start times
array [NODE,CORE] of var 0..1: sel;             % 1 when the ws is selected to execute the job
var lb..total: end;                            % total end time, the variable representing the makespan
% aux variable used only to hold the selected initial time and duration for each job
% this will ease the DAG precedence constraint
% used[j,1]: initial time
% used[j,2]: duration
array [NODE,1..2] of var 0..total: used;

% a job runs on a single workstation
constraint forall(i in NODE)(
             sum(row(sel,i)) == 1);

% ok for dag deadline > period
% constraint %% utilization constraint
%     forall(j in CORE) (
%         sum(i in NODE) (
%             int2float(sel[i,j]) * U[i]) <= 1.0
%     );

constraint %% check utilization constraint for the set of unrelated tasks that can run concurrently
    max(u in 1..N_UNREL_ROWS)(
        sum(t in 1..N_UNREL_COLS)(U[unrelated_node[u,t]])
    ) <= 1.0;

constraint %% main constraint to check the DAG deadline
    forall(j in CORE) (
        forall(i in NODE) (
            % replace this to minimize makespan
            sel[i,j]*(s[i,j] + D[i]) <= Ddag)
            %sel[i,j]*(s[i,j] + D[i]) <= end)
    );

%% enable this to also minimize the makespan
%constraint  end <= Ddag;

% helper constraint to specify the DAG precedence
constraint
    forall(i in NODE)(
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
            var NODE: sj = to_enum(NODE,E[i,1]); % source job
            var NODE: tj = to_enum(NODE,E[i,2]); % target job
            } in 
            precedence (used[sj,1], used[sj,2], used[tj,1])
    );

predicate no_overlap(var int: start0, var int: duration0,
                     var int: start1, var int: duration1) =
  start0 + duration0 <= start1 \/ start1 + duration1 <= start0;
  
constraint %% ensure no overlap of tasks
    forall(j in CORE) (
        forall(i,k in NODE where i < k) (
            no_overlap(s[i,j],T[i], s[k,j],T[k])
        )
    );

% replace this commentto minimize makespan
% solve minimize end;
solve satisfy;

output ["end = \(end)\n"] ++
       ["ub = \(total)\n"] ++
       ["lb = \(lb)\n"] ++
       ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in NODE, j in CORE 
       ] ++
       ["start_time, duration\n"] ++ 
       [ show_int(digs,used[i,j]) ++ " " ++ 
         if j == 2 then "\n" else "" endif |  
         i in NODE, j in 1..2 
       ];
