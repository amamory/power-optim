% The set of nodes (NODE) with  precedence constraints modeled as a DAG,
% (note: except for the first and the last nodes, the other nodes represents tasks)
% must be schedule on any of hw processing units (PU) represented by set CORE|.
% The tasks have a relative deadline (D) and a utilization (U). 

% Objective:
% - minimize the max utilization among the PU (U_max)

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
array [NODE] of int: D;                        % task deadline
array [1..N_NODES] of float: U;                % PU utilization per task
array [1..N_EDGES,1..2] of 1..N_NODES: E;      % the DAG edges: source node to target node

array [1..N_UNREL_ROWS,1..N_UNREL_COLS] of 1..N_NODES: unrelated_node;

% AUX CONTANTS
int: total=300;
% calculate the max number of digits just for printing the output
int: digs = ceil(log(10.0,int2float(total)));  
CORE: last_core = max(CORE);
NODE: last_node = max(NODE);
float: total_utillization = sum(i in 1..N_NODES)(U[i]);

% VARIABLES
array [NODE] of var 0..total: s;           % start times for each task
array [NODE,CORE] of var 0..1: sel;        % 1 when the ws is selected to execute the task
var 0.0..total_utillization: U_max;        % the objective function
% utilization per PU, used only for printing the output
array [CORE] of var 0.0..total_utillization: pu_utilization; 

% a job runs on a single PU
constraint forall(i in NODE)(
             sum(row(sel,i)) == 1);

%% check utilization constraint for the set of unrelated tasks that can run concurrently
constraint 
    max(u in 1..N_UNREL_ROWS)(
        sum(t in 1..N_UNREL_COLS)(U[unrelated_node[u,t]])
    ) <= 1.0;

% Minize the max utilization of the cores.
% Sum the utilization for each core and minimizes the maximal utilization, 
% promoting some kind of load balancing among the cores
constraint 
    forall(c in CORE)(
        pu_utilization[c] = sum(t in NODE)(sel[t,c]*U[t])
    );

constraint 
    max(c in CORE)(pu_utilization[c]) <= U_max;

% the starting time of the dummy final task + its deadline must be <= the dag deadline
constraint ((s[last_node] + D[last_node]) <= Ddag);

% the starting time of the dummy first task must be zero
constraint (s[to_enum(NODE,1)] == 0);

% job0 has precedence over job1
predicate precedence(var int: start0, var int: duration0, var int: start1) =
  start0 + duration0 <= start1 ;

constraint % the DAG precedence
    forall(i in EDGE) (
        let { 
            var NODE: source = to_enum(NODE,E[i,1]); % source job
            var NODE: target = to_enum(NODE,E[i,2]); % target job
            } in 
            precedence (s[source], D[source], s[target])
    );

solve minimize U_max;
% solve satisfy;

output ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last_core then "\n" else "" endif |
         i in NODE, j in CORE 
       ] ++
       ["PU utilization: "] ++
       [ show_float(digs,2,pu_utilization[i]) ++ " " |   
         i in CORE 
       ] ++
       ["\nstart times: "] ++
       [ show_int(digs,s[i]) ++ " " |   
         i in NODE 
       ];
