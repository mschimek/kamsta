[![DOI](https://zenodo.org/badge/602074500.svg)](https://zenodo.org/badge/latestdoi/602074500)

# KaMSTa

Implementations of our distributed **M**inimum **S**panning **T**ree (MST) algorithms which we present in our paper:

_P. Sanders and M. Schimek. Engineering Massively Parallel MST Algorithms._ to appear in 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS).

If you use this code in the context of an academic publication, please cite the [freely accessible postprint](https://arxiv.org/abs/2302.12199):
```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.12199,
  doi = {10.48550/ARXIV.2302.12199},
  
  url = {https://arxiv.org/abs/2302.12199},
  
  author = {Sanders, Peter and Schimek, Matthias},
  
  keywords = {Distributed, Parallel, and Cluster Computing (cs.DC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Engineering Massively Parallel MST Algorithms},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Compiling

To compile the code use the following instructions:
```
  git clone --recursive <https://github.com/mschimek/kamsta.git|git@github.com:mschimek/kamsta.git>
  cmake -B build -DCMAKE_BUILD_TYPE=Release [-DUSE_EXPLICIT_INSTANTIATION=ON]
  cmake --build build --parallel
```

## Usage

To execute the code use the following instructions:

```sh
  export OMP_NUM_THREADS=<number threads>
  mpiexec -n <number mpi processes> ./build/benchmarks/mst_benchmarks [kamsta parameters (--help for help])
```
See the [evaluation directory](https://github.com/mschimek/kamsta/tree/main/evaluation) for generating preconfigured parameter settings similar to the ones we used in our experiments.

Furthermore, it is important that `OMP_NUM_THREADS` matches the kamsta parameter `--threads`.
If you use multithreading, you should consider allocating enough CPUs per MPI process to avoid performance problems.


## Dependencies
Apart from the included submodules we use `OpenMP` and `Thread Building Blocks`(TBB).


## Notes
Note that due to restrictions in the graph generator, we tested our implementation only with number of threads and number of mpi processes being powers of two.
We tested our implementation with GCC 10/11/12 and OpenMPI 4.0.

The code version used in our paper can be found in [this release](https://github.com/mschimek/kamsta/releases/tag/v0.2).


If you encounter problems, feel free to contact us or open a pull request.
