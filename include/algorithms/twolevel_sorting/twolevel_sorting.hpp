#pragma once

#include "RQuick/RQuick.hpp"
#include <mpi/alltoall_combined.hpp>
#include <mpi/broadcast.hpp>
#include <mpi/grid_communicators.hpp>
#include <mpi/scan.hpp>
#include <mpi/type_handling.hpp>
#include <random>
#include <util/timer.hpp>
#include <util/utils.hpp>
#include <vector>

namespace hybridMST {
namespace sorting_internal {
template <typename Container,
          typename ValueType = typename Container::value_type>
std::vector<ValueType> get_samples(Container& elements,
                                   std::size_t num_samples) {
  mpi::MPIContext ctx;
  std::mt19937 gen(ctx.rank() + elements.size() % 1000);
  std::uniform_int_distribution<std::size_t> sample_idx(0, elements.size() - 1);
  if (elements.empty())
    return {};
  std::vector<ValueType> samples(num_samples);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    samples[i] = elements[sample_idx(gen)];
  }
  return samples;
}

template <typename T, typename Comp>
std::vector<T>
select_splitters_level_1(const mpi::PowerTwoGridCommunicators& comm,
                         std::vector<T>& local_samples, Comp&& comp) {
  mpi::MPIContext world_ctx;
  mpi::TypeMapper<T> tm;
  const std::size_t num_columns = comm.num_columns();
  int tag = 100000;
  std::mt19937_64 gen(world_ctx.rank());
  get_timer().start_phase_measurement("partition_splitters_rquick");

  const std::size_t num_bytes_sent = local_samples.size() * sizeof(T);
  RQuick::sort(tm.get_mpi_datatype(), local_samples, tag, gen,
               world_ctx.communicator(), comp);
  const std::size_t num_bytes_recv = local_samples.size() * sizeof(T);
  get_communication_tracker().add_volume(num_bytes_sent, num_bytes_recv);
  get_timer().stop_phase_measurement("partition_splitters_rquick");
  get_timer().start_phase_measurement("partition_splitters_rest");
  const std::size_t num_global_samples =
      mpi::allreduce_sum(local_samples.size());
  const double num_normalized_local_samples =
      static_cast<double>(num_global_samples) / num_columns;
  const std::size_t global_start_idx =
      mpi::exscan_sum(local_samples.size(), world_ctx, 0ul);
  std::vector<T> splitters;
  auto is_sample_present = [&](std::size_t idx) {
    return global_start_idx <= idx &&
           idx < global_start_idx + local_samples.size();
  };
  // SEQ_EX(world_ctx, PRINT_VAR(num_columns););

  for (std::size_t i = 1; i < num_columns; ++i) {
    const std::size_t splitter_idx = i * num_normalized_local_samples;
    if (is_sample_present(splitter_idx)) {
      const std::size_t local_splitter_idx = splitter_idx - global_start_idx;
      splitters.push_back(local_samples[local_splitter_idx]);
    }
  }
  get_timer().stop_phase_measurement("partition_splitters_rest");
  get_timer().start_phase_measurement("partition_splitters_rest_allgather");
  const auto& column_ctx = comm.get_col_ctx();
  const auto& row_ctx = comm.get_row_ctx();
  splitters = mpi::allgatherv(splitters, column_ctx);
  if (column_ctx.rank() == 0) {
    splitters = mpi::allgatherv(splitters, row_ctx);
  } else {
    splitters = mpi::allgatherv(
        splitters,
        row_ctx); // only for consistency in communication volume tracking
    splitters.clear();
  }
  // SEQ_EX(world_ctx, PRINT_VAR(splitters););
  mpi::bcast(splitters, 0, column_ctx);
  get_timer().stop_phase_measurement("partition_splitters_rest_allgather");
  MPI_ASSERT_(splitters.size() == (row_ctx.size() - 1), "");
  return splitters;
}

inline std::vector<PEID> pseudo_random_offsets(std::size_t num_elements,
                                               PEID rank,
                                               std::size_t num_rows) {
  num_elements = std::min(num_elements, static_cast<std::size_t>(1000ul));
  std::vector<PEID> offsets(num_elements);
  std::mt19937 gen(rank * 2 + 1);
  std::uniform_int_distribution<std::size_t> offset_distribution(0,
                                                                 num_rows - 1);
  for (auto& offset : offsets) {
    offset = offset_distribution(gen);
  }
  return offsets;
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
void firstlevel_partition(Container& elements, Comp comp = Comp{}) {
  using T = typename Container::value_type;
  mpi::MPIContext world_ctx;
  const auto& grid_communicators = mpi::get_power_two_grid_communicators();
  get_timer().start_phase_measurement("partition_samples");
  const std::size_t num_needed_splitters =
      16 * std::log2(grid_communicators.num_columns());
  const std::size_t num_local_samples = std::max(num_needed_splitters, 10ul);
  auto samples = sorting_internal::get_samples(elements, num_local_samples);
  get_timer().stop_phase_measurement("partition_samples");
  // get_timer().start_phase_measurement("partition_splitters");
  auto splitters = sorting_internal::select_splitters_level_1(
      grid_communicators, samples, comp);
  // SEQ_EX(world_ctx, PRINT_VECTOR(splitters););
  //  get_timer().stop_phase_measurement("partition_splitters");
  const auto offsets = pseudo_random_offsets(elements.size(), world_ctx.rank(),
                                             grid_communicators.num_rows());
  const std::size_t num_rows = grid_communicators.num_rows();
  auto filter = False_Predicate{};
  auto transformer = [&](const T& t, const std::size_t&) { return t; };
  auto dst_computer = [&](const T& t, const std::size_t& i) {
    const auto it =
        std::upper_bound(splitters.begin(), splitters.end(), t, comp);
    const std::size_t offset_entry = i % offsets.size();
    return (std::distance(splitters.begin(), it) * num_rows) +
           offsets[offset_entry];
  };
  // auto partitioned_data = mpi::twopass_alltoallv_openmp_special(
  //     elements, filter, transformer, dst_computer, ctx.size(),
  //     ctx.threads_per_mpi_process());
  auto partitioned_data =
      mpi::alltoall_combined(std::move(elements), filter, transformer, dst_computer);
  auto elems_per_group = allreduce_sum(partitioned_data.buffer.size(),
                                       grid_communicators.get_col_ctx());
  // SEQ_EX(world_ctx, if(twolevel_comm.get_col_ctx().rank() == 0) {
  // PRINT_VAR(elems_per_group);});
  elements = std::move(partitioned_data.buffer);
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
void secondlevel_partition(Container& elements, Comp comp = Comp{}) {
  using T = typename Container::value_type;
  const auto& grid_communicators = mpi::get_power_two_grid_communicators();

  auto samples = get_samples(elements, grid_communicators.get_col_ctx());
  auto splitters =
      select_splitters(samples, comp, grid_communicators.get_col_ctx());
  auto filter = False_Predicate{};
  auto transformer = [&](const T& t, const std::size_t&) { return t; };
  auto dst_computer = [&](const T& t, const std::size_t&) {
    const auto it =
        std::upper_bound(splitters.begin(), splitters.end(), t, comp);
    return std::distance(splitters.begin(), it);
  };
  auto partitioned_data = mpi::twopass_alltoallv_openmp_special(
      std::move(elements), filter, transformer, dst_computer,
      grid_communicators.get_col_ctx());
  elements = std::move(partitioned_data.buffer);
}
} // namespace sorting_internal

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
void twolevel_partition(Container& elements, Comp comp = Comp{}) {
  using T = typename Container::value_type;
  sorting_internal::firstlevel_partition(elements, comp);
  sorting_internal::secondlevel_partition(elements, comp);
  mpi::MPIContext ctx;
  // SEQ_EX(ctx, PRINT_VAR(elements.size()););
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
auto twolevel_sorting(Container& elements, Comp comp = Comp{}) {
  twolevel_partition(elements, comp);
  ips4o::parallel::sort(elements.begin(), elements.end(), comp);
  return elements;
}

} // namespace hybridMST
