#pragma once

#include <random>

#include "RQuick/RQuick.hpp"

#include "mpi/allgather.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/context.hpp"
#include "mpi/scan.hpp"
#include "mpi/type_handling.hpp"
#include "util/macros.hpp"
#include "util/utils.hpp"

namespace hybridMST {
template <typename Container,
          typename ValueType = typename Container::value_type>
std::vector<ValueType> get_samples(const Container& elements,
                                   std::size_t num_samples,
                                   mpi::MPIContext ctx = mpi::MPIContext{}) {
  std::mt19937 gen(ctx.rank() + elements.size() % 1000);
  std::uniform_int_distribution<std::size_t> sample_idx(0, elements.size() - 1);
  if (elements.empty())
    return {};
  std::vector<ValueType> samples(num_samples);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    samples[i] = elements[sample_idx(gen)];
  }
  // SEQ_EX(ctx, PRINT_VECTOR(samples););
  return samples;
}

template <typename Container,
          typename ValueType = typename Container::value_type>
std::vector<ValueType> get_samples(Container& elements,
                                   mpi::MPIContext ctx = mpi::MPIContext{}) {
  const std::size_t oversampling_ratio = 32;
  const std::size_t num_samples =
      1 + (oversampling_ratio * std::log2(ctx.size()));
  return get_samples(elements, num_samples, ctx);
}

template <typename T, typename Comp>
std::vector<T> select_splitters(std::vector<T>& local_samples, Comp&& comp,
                                mpi::MPIContext ctx = mpi::MPIContext()) {
  mpi::TypeMapper<T> tm;
  int tag = 100000;
  std::mt19937_64 gen(ctx.rank());
  //  const auto& count = get_timer().get_phase_add_count();
  //  get_timer().add_phase("partition_local_sample_size", count,
  //                        local_samples.size(),
  //                        {Timer::DatapointsOperation::ID});
  //  get_timer().start_phase_measurement("partition_splitters_rquick");
  RQuick::sort(tm.get_mpi_datatype(), local_samples, tag, gen,
               ctx.communicator(), comp);
  // const auto& count2 = get_timer().get_phase_add_count();
  //  get_timer().add_phase("partition_local_sample_size_res", count2,
  //                        local_samples.size(),
  //                        {Timer::DatapointsOperation::ID});
  //  get_timer().stop_phase_measurement("partition_splitters_rquick");
  //  get_timer().start_phase_measurement("partition_splitters_rest");
  const std::size_t num_global_samples =
      mpi::allreduce_sum(local_samples.size(), ctx);
  const std::size_t num_normalized_local_samples =
      num_global_samples / ctx.size();
  const std::size_t global_start_idx =
      mpi::exscan_sum(local_samples.size(), ctx, 0ul);
  std::vector<T> splitters;
  auto is_sample_present = [&](std::size_t idx) {
    return global_start_idx <= idx &&
           idx < global_start_idx + local_samples.size();
  };

  for (std::size_t i = 1; i < static_cast<std::size_t>(ctx.size()); ++i) {
    const std::size_t splitter_idx = i * num_normalized_local_samples;
    if (is_sample_present(splitter_idx)) {
      const std::size_t local_splitter_idx = splitter_idx - global_start_idx;
      splitters.push_back(local_samples[local_splitter_idx]);
    }
  }
  //  get_timer().stop_phase_measurement("partition_splitters_rest");
  //  get_timer().start_phase_measurement("partition_splitters_rest_allgather");
  splitters = mpi::allgatherv(splitters, ctx);
  //  get_timer().stop_phase_measurement("partition_splitters_rest_allgather");
  MPI_ASSERT_(splitters.size() == (ctx.size() - 1), "");
  return splitters;
}

template <typename T, typename Comp>
Weight select_pivot(std::vector<T>& local_samples, Comp&& comp) {
  mpi::MPIContext ctx;
  mpi::TypeMapper<T> tm;
  int tag = 100001;
  std::mt19937_64 gen(ctx.rank());
  const std::size_t num_bytes_sent = local_samples.size() * sizeof(T);
  RQuick::sort(tm.get_mpi_datatype(), local_samples, tag, gen,
               ctx.communicator(), comp);
  const std::size_t num_bytes_recv = local_samples.size() * sizeof(T);
  get_communication_tracker().add_volume(num_bytes_sent, num_bytes_recv);

  const std::size_t num_global_samples =
      mpi::allreduce_sum(local_samples.size());
  const std::size_t global_start_idx =
      mpi::exscan_sum(local_samples.size(), ctx, 0ul);
  auto is_sample_present = [&](std::size_t idx) {
    return global_start_idx <= idx &&
           idx < global_start_idx + local_samples.size();
  };
  const std::size_t pivot_index = num_global_samples / 2;
  const bool is_root = is_sample_present(pivot_index);
  if (is_root) {
    const std::size_t local_pivot_idx = pivot_index - global_start_idx;
    Weight w = local_samples[local_pivot_idx].get_weight();
    return mpi::allreduce_min(w);
  }
  return mpi::allreduce_min(WEIGHT_INF);
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
auto partition(Container& elements, Comp comp = Comp{},
               mpi::MPIContext ctx = mpi::MPIContext{}) {
  using T = typename Container::value_type;
  // get_timer().start_phase_measurement("partition_samples");
  auto samples = get_samples(elements, ctx);
  // get_timer().stop_phase_measurement("partition_samples");
  // get_timer().start_phase_measurement("partition_splitters");
  auto splitters = select_splitters(samples, comp, ctx);
  // get_timer().stop_phase_measurement("partition_splitters");
  auto filter = False_Predicate{};
  auto transformer = [&](const T& t, const std::size_t&) { return t; };
  auto dst_computer = [&](const T& t, const std::size_t&) {
    const auto it =
        std::upper_bound(splitters.begin(), splitters.end(), t, comp);
    return std::distance(splitters.begin(), it);
  };
  // auto partitioned_data = mpi::twopass_alltoallv_openmp_special(
  //     elements, filter, transformer, dst_computer, ctx.size(),
  //     ctx.threads_per_mpi_process());
  auto partitioned_data =
      mpi::alltoall_combined(elements, filter, transformer, dst_computer);
  return partitioned_data.buffer;
}

} // namespace hybridMST
