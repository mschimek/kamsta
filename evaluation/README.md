# Running Experiments

The scripts in this directory provide some assistances with running our algorithms.
YAML files are used to describe a specific setup. These consist conceptually of three parts:
  - number of workers (MPI processes/threads) to use
  - graphs to use as input
  - paramter configuration
  
If multiple options are given per entry, jobs for all possible combinations are created.
A call to `generate_jobs.py` will create a new directory containing the according compute job files.

```sh
python ./generate_jobs.py --experiment_parent_directory <path to directory where experiment directory will be created> \
                          --experiment_name <name of the experiment directory to be created>
                          --experiment_suite <YAML file describing the experiment>
                          --platform <shared_memory|supermuc|horeka>
```

See `experiment_suites` for YAML files used in our experiments.

For real-world graphs we use a binary graph format which simply stores a (possibly unsorted) edge list representation of the graphs.
Each edge is a 9 Byte record (4 Byte source/destination vertex, 1 Byte edge weight).
