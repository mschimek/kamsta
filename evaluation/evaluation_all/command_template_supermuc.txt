echo "[LOG] Starting ${jobname}"
start=$$(date +%s.%N)

PARLAY_NUM_THREADS=${num_threads} OMP_NUM_THREADS=${num_threads} mpiexec  \
  -n ${num_mpi_procs}                                                     \
  --map-by ppr:${num_mpi_procs_per_node}                                  \
  --map-by node:PE=${num_cpus_per_mpi_proc}                               \
  --bind-to core                                                          \
  ${cmd}                                       

end=$$(date +%s.%N)
diff=$$(echo "$end - $start" | bc)
echo "[LOG] Finished ${jobname} in $${diff} seconds"
sleep ${sleep_duration}


