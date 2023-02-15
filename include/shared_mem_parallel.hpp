#pragma once

#include <omp.h>

namespace hybridMST {
template <typename F>
inline void parallel_for(std::size_t begin, std::size_t end, F&& f) {
#pragma omp parallel for schedule(static)
  for (std::size_t i = begin; i < end; ++i) {
    f(i);
  }
}

template <typename F>
inline void parallel_for(std::size_t begin, std::size_t end,
                         std::size_t granularity, F&& f) {
#pragma omp parallel for schedule(static, granularity)
  for (std::size_t i = begin; i < end; ++i) {
    f(i);
  }
}
} // namespace hybridMST
