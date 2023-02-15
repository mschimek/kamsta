#pragma once

#include "mpi/alltoall.hpp"
#include "mpi/twolevel_alltoall.hpp"

namespace hybridMST::mpi {

template <typename Container, typename FilterOut>
bool use_two_level_alltoall(const Container& data, FilterOut&& filter_out) {
  mpi::MPIContext ctx;
  std::size_t num_remaining_elements = 0;
#pragma omp parallel for reduction(+ : num_remaining_elements)
  for (std::size_t i = 0; i < data.size(); ++i) {
    num_remaining_elements += !filter_out(data[i], i);
  }
  num_remaining_elements = mpi::allreduce_sum(num_remaining_elements);
  const std::size_t threshold_avg_num_elems_per_msg = 500;
  const std::size_t ctx_size = ctx.size();
  const bool use_two_level_approach =
      (num_remaining_elements / (ctx_size * ctx_size)) <=
      threshold_avg_num_elems_per_msg;
  return use_two_level_approach;
}
template <typename Container, typename Filter, typename Transformer,
          typename DstCalculator>
auto alltoall_combined(Container&& data, Filter&& filter,
                       Transformer&& transformer,
                       DstCalculator&& dstCalculator) {

  if (use_two_level_alltoall(data, std::forward<Filter>(filter))) {
    return two_level_alltoall_extract(
        std::forward<Container>(data), std::forward<Filter>(filter),
        std::forward<Transformer>(transformer),
        std::forward<DstCalculator>(dstCalculator));
  }
  mpi::MPIContext ctx;
  return twopass_alltoallv_openmp_special(
      std::forward<Container>(data), std::forward<Filter>(filter),
      std::forward<Transformer>(transformer),
      std::forward<DstCalculator>(dstCalculator), ctx);
}
} // namespace hybridMST::mpi
