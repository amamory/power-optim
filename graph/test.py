
## basic test cases
## - test for # of islands
## - test for # of npus
## - test for # of dags
## - test for # of tasks
## - test minimal platform: 1 island, 1 freq
## - test minimal sw: 1 dag, 1 task

## parallel test cases
## - define the sw and hw inputs
## - test with 1 proc
## - test with 3 procs. expect the same results as the previous one

## 1 island result test cases
## - test platform: 1 island, 3 freq
## - test minimal sw: 1 dag, 1 task
## - a case where it is only feasible at the minimal freq
## - for the previous case, increase the dag deadline such that it is feasible at all freqs. expect the same solution as the previous

## unrelated test case
## - 4 tasks in parallel. no depedency
## - case study w a solution
## - case study w that fails in the utilization constraint

## high utilization test case
## - devise a case where even the low capacity islands are loaded with the highest freqs