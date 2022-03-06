%include params;

%https://en.wikipedia.org/wiki/Open-shop_scheduling
%https://www.minizinc.org/doc-2.5.3/en/downloads/jobshop.mzn
%https://github.com/MiniZinc/minizinc-benchmarks/blob/master/openshop/openshop.mzn
     
enum JOB;
enum CORE;
CORE: last = max(CORE);
array [JOB,CORE] of int: d;                   % task durations

% BOUNDS FOR MAKESPAN 
% UPPER BOUND: since a job runs on a single workstation, lets be pessimistic and assume that the longest duration will be chosen for each pair job/core
% i.e., for each job, choose the longest duration
int: total = sum(i in JOB)(max(row(d,i)));  % total duration, upper bound for makespan
% LOWER BOUND: lets be optimistic and choose the shortest durations for each Job, sum their durations, and divide by the number of cores
int: lb = ceil(int2float(sum(i in JOB)(min(row(d,i)))) / int2float(card(CORE)));  % total duration, lower bound for makespan

% AUX CONTANTS
int: digs = ceil(log(10.0,int2float(total)));      % calculate the max number of digits just for printing the output

% VARIABLES
array [JOB,CORE] of var 0..total: s;           % start times
array [JOB,CORE] of var 0..1: sel;             % 1 when the ws is selected to execute the job
var lb..total: end;                              % total end time, the variable representing the makespan

% a job runs on a single workstation
constraint forall(i in JOB)(
             sum(row(sel,i)) == 1);

% constraint %% ensure the tasks occur in sequence
%     forall(i in JOB) (
%         forall(j in CORE where j < last) 
%             (s[i,j] + d[i,j] <= s[i,enum_next(CORE,j)]) /\
%         s[i,last] + d[i,last] <= end
%     );

constraint %% minimize the makespan
    forall(j in CORE) (
        forall(i in JOB) (
            sel[i,j]*(s[i,j] + d[i,j]) < end)
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
       [ show_int(digs,s[i,j]*sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in CORE ] ++
       ["------------\n"] ++
       [ show_int(digs,sel[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in CORE ] ++
       ["------------\n"] ++ 
       [
           ( if fix(sel[i,j]) = 1 then show_int(digs,j) ++ ": " ++ show_int(digs,s[i,j]) ++ " " ++ show_int(digs,d[i,j]) ++ "\n" else "" endif ) 
           |i in JOB, j in CORE
       ] ;
         