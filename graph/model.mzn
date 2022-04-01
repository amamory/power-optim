

% MAIN DEFINITIONS
int : N_TASKS;
int : N_ISLANDS;
int : N_DAGS;
int : N_EDGES;
int : D;
enum TASK = anon_enum(N_TASKS);
enum DAG = anon_enum(N_DAGS);
enum ISLAND = anon_enum(N_ISLANDS);
enum EDGE = anon_enum(N_EDGES);
array [ISLAND,1..4] of int: I;                % array of islands
array [ISLAND] of 0.0 .. 1.0: I_cap;          % the capacity of each island
array [TASK,1..2] of int: T;                  % array of tasks
array [DAG,1..3] of int: G;                   % array of DAGs
array [1..N_EDGES,1..2] of 0..N_TASKS-1: E;   % edges of the DAG !!!! TODO: currently considering only one DAG !!!!

% BOUNDS
int: total = sum(i in TASK)(max(row(T,i)));
int: max_int = max(total,ub_p);

% sum the number of cores in each island
int: total_cores = sum(i in I)(i[0]);
array [1..N_TASKS,1..total_cores] of int: C;    % WCET for each task assuming each core. Note that the freq for each core/island is fixed

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(max_int)));  % calculate the max number of digits just for printing the output

% VARIABLES
array [1..N_TASKS,1..total_cores] of var 0..1: sel;  % 1 when the ws is selected to execute the task
array [1..N_TASKS] of 0..total: f;                   % finishing times of all tasks

% #####################################
% UTILIZATION CONSTRAINTS
% #####################################

% a core hosts tasks up to its maximum utilization of 1.0
constraint 
    forall (g in G)(
        let { 
            var DAG: p = g[1]; % DAG activation period
        } in        
        forall(t in 1..N_TASKS)(
            forall(c in 1..total_cores)(
                sum(int2float(sel[t][c]*C[t][c])/int2float(p)) <= 1.0
            )
            % TODO replace p by relative deadline
        )
    );
% #####################################
% PRECEDENCE CONSTRAINTS
% #####################################

% finishing time for the initial node equals to its deadline. always identified as 'node 0'
% wcet 3 10 
% capacity 0.2 
% freq 50 
% ref freq 1000
% = 31.000000000000004
constraint f[0] = T[0][3];

constraint 
    forall(i in 1..TASK)(
        % e[0]:  source node
        % e[1]:  target node
        % T[i][3]: deadline for task T_i
        f[i] = T[i][3] + max(e in E where e[1] = i)( f[e[0]] )
    );

% the finishing time of the last node
% the deadline constraint
constraint f[TASK] <= D;

% #####################################
% OTHER CONSTRAINTS
% #####################################

% a task runs on a single core
constraint 
    forall(i in TASK)(
        sum(row(sel,i)) == 1
    );



% solve minimize end; % makespan
solve satisfy;

output ["end = \(end)\n"] ++
       ["ub = \(total)\n"] ++
       ["lb = \(lb)\n"] ++
       ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == total_cores then "\n" else "" endif |
         i in TASK, j in 1..total_cores 
       ];
