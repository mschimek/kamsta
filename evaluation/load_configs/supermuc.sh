#!/bin/bash
load_gcc_config() {
  module purge
    module load admin/1.0
    module load tempdir/1.0
    module load lrz/1.0
    module load spack/21.1.1
    module load cmake/3.16.5

    module load gcc/10.2.0
    module load intel-tbb/2020.3
    module load openmpi/4.0.4-gcc8
    module load boost/1.73.0-gcc8
}

