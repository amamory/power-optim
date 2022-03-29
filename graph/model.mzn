

% MAIN DEFINITIONS
int : N_TASKS;
int : N_ISLANDS;
int : N_DAGS;
int : D;
enum TASK = anon_enum(N_TASKS);
enum DAG = anon_enum(N_DAGS);
enum ISLAND = anon_enum(N_ISLANDS);
array [ISLAND,1..4] of int: I;                % array of islands
array [ISLAND] of 0.0 .. 1.0: I_cap;          % the capacity of each island
array [TASK,1..2] of int: T;                  % array of tasks
array [DAG,1..3] of int: G;                   % array of DAGs

% BOUNDS
int: total = sum(i in TASK)(max(row(T,i)));
int: max_int = max(total,ub_p);

% sum the number of cores in each island
int: total_cores = sum(i in I)(i[0]);

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(max_int)));  % calculate the max number of digits just for printing the output

% VARIABLES
array [TASK,1..total_cores] of var 0..1: sel;  % 1 when the ws is selected to execute the task
var lb..total: end;                            % total end time, the variable representing the makespan

% deadline constraint: makespan (end) must be shorter or equal to the deadline (D)
constraint end <= D;

% a task runs on a single core
constraint 
    forall(i in TASK)(
        sum(row(sel,i)) == 1
    );

constraint %% define the makespan
    forall(j in ISLAND) (
        forall(i in TASK) (
            sel[i,j]*(s[i,j] + d[i,j]) <= end)
    );

% a core hosts tasks up to its maximum utilization of 1.0
constraint 
    forall (g in G)(
        let { 
            var DAG: p = g[1]; % DAG activation period
        } in        
        forall(p in 1..total_cores)(
            let { 
                array of 0..1: A = col(sel,p); % array of tasks assigned to core p
            } in              
            sum(t in TASK, a in A)(int2float(a*t[3]/int2float(p)) <= max_util);
        )
    );

% helper constraint to specify the DAG precedence
constraint
    forall(i in TASK)(
        forall(j in ISLAND where sel[i,j] == 1) (
            used[i,1] == s[i,j] /\ used[i,2] == d[i,j] /\ used[i,3] == p[i,j] 
        )
    );

% task0 has precedence over task1
predicate precedence(var int: start0, var int: duration0, var int: start1) =
  start0 + duration0 <= start1 ;

constraint % the DAG precedence
    forall(i in EDGE) (
        let { 
            var TASK: sj = to_enum(TASK,e[i,1]); % source task
            var TASK: tj = to_enum(TASK,e[i,2]); % target task
            } in 
            precedence (used[sj,1], used[sj,2], used[tj,1])
    );

predicate no_overlap(var int: start0, var int: duration0,
                     var int: start1, var int: duration1) =
  start0 + duration0 <= start1 \/ start1 + duration1 <= start0;
  
constraint %% ensure no overlap of tasks
    forall(j in ISLAND) (
        forall(i,k in TASK where i < k) (
            no_overlap(s[i,j],d[i,j], s[k,j],d[k,j])
        )
    );

% choose one of the possible optimization functions
solve minimize end; % makespan


output ["end = \(end)\n"] ++
       ["ub = \(total)\n"] ++
       ["lb = \(lb)\n"] ++
       ["selected cores:\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == total_cores then "\n" else "" endif |
         i in TASK, j in 1..total_cores 
       ];
