# KaMSTa

Implementations of our distributed **M**inimum **S**panning **T**ree (MST) algorithms.

## Compiling

To compile the code use the following instructions:
```
  git submodule update --init --recursive
  cmake -B build -DCMAKE_BUILD_TYPE=Release [-DUSE_EXPLICIT_INSTANTIATION=ON]
  cmake --build build --parallel
```

## Usage

To compile the code use the following instruction
```
  export OMP_NUM_THREADS=<number threads>
  mpiexec -n <number mpi processes> ./build/benchmarks/mst_benchmarks [kamsta parameters (--help for help])
```

Furthermore, it is important that `OMP_NUM_THREADS` matches the kamsta parameter `--threads`.
If you use multithreading, you should consider allocating enough CPUs per MPI process to avoid performance problems.

## Notes
Note that due to restrictions in the graph generator, we tested our implementation only with number of threads and number of mpi processes being power of two.
We tested our implementation with GCC 10/11/12 and OpenMPI 4.0.

If you encounter problems, feel free to contact us or open a pull request.

## Dependencies
Apart from the included submodules we use `OpenMP` and `Thread Building Blocks`(TBB).
