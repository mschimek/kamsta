#pragma once

#include <omp.h>

namespace hybridMST {
  template <typename F>
  inline void parallel_for(std::size_t begin, std::size_t end, F&& f) {
    #pragma omp parallel for
    for(std::size_t i = begin; i < end; ++i) {
      f(i);
    }
  }

  template <typename F>
  inline void parallel_for(std::size_t begin, std::size_t end, std::size_t granularity, F&& f) {
    parallel_for(begin, end, std::forward<F>(f));
  }
}
