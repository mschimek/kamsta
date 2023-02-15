#!/bin/bash


trap exit SIGINT
echo_and_run() { echo "\$ $*"; "$@"; }

# parameter handling
num_workers=$1 
if [ -z "$num_workers" ]
then
  num_workers=8
fi

# fix setup
algorithms=("hybridBoruvka" "filter_hybridBoruvka")
graphs=("RMAT" "GNM" "RGG_2D" "RHG")
threads=(4)
logfile="out.txt"
logfile_errors="out_err.txt"


> ${logfile}
> ${logfile_errors}

for i in ${!algorithms[@]}; do
  algorithm=${algorithms[$i]}
  for j in ${!graphs[@]}; do
    graph=${graphs[$j]}
    for k in ${!threads[@]}; do
      num_threads=${threads[$k]}
      num_processes=$((num_workers / num_threads))
      echo "OMP_NUM_THREADS=${num_threads} mpiexec -n ${num_processes} ../build/benchmarks/mst_benchmarks \
          --algorithm $algorithm        \
          --graphtype ${graph}          \
          --threads ${num_threads}      \
          --weak_scaling_level 1        \
          --checks                      \
          --log_n 12                    \
          --log_m 15" >> ${logfile}
      OMP_NUM_THREADS=${num_threads} mpiexec -n ${num_processes} ../build/benchmarks/mst_benchmarks \
          --algorithm $algorithm        \
          --graphtype ${graph}          \
          --threads ${num_threads}      \
          --weak_scaling_level 1        \
          --checks                      \
          --log_n 12                    \
          --log_m 15 >> ${logfile} 2>>${logfile_errors}
      if [ -s ${logfile_errors} ]; then
        echo "Some errors occured."
      fi
    done
  done
done

if [ -s ${logfile_errors} ]; then
  echo "Some errors occured during the test runs. See ${logfile_errors} and ${logfile} for details."
  exit 1
fi
