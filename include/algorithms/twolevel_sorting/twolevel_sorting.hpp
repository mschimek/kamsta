#pragma once

#include <random>
#include <vector>

#include "RQuick/RQuick.hpp"

#include "mpi/alltoall_combined.hpp"
#include "mpi/broadcast.hpp"
#include "mpi/scan.hpp"
#include "mpi/twolevel_columnmajor_communicator.hpp"
#include "mpi/type_handling.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

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
select_splitters_level_1(const mpi::TwoLevelColumnMajorCommunicator& comm,
                         std::vector<T>& local_samples, Comp&& comp) {
  mpi::MPIContext world_ctx;
  mpi::TypeMapper<T> tm;
  const std::size_t num_columns = comm.num_cols();
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
  splitters = mpi::allgatherv(splitters, column_ctx);
  splitters = mpi::row_wise_allgatherv_on_column_data(splitters, comm);
  // SEQ_EX(world_ctx, PRINT_VAR(splitters););
  get_timer().stop_phase_measurement("partition_splitters_rest_allgather");
  //MPI_ASSERT_(splitters.size() == (row_ctx.size() - 1), "");
  return splitters;
}

inline std::vector<PEID> pseudo_random_offsets(std::size_t num_elements,
                                               PEID rank) {
  num_elements = std::min(num_elements, static_cast<std::size_t>(1000ul));
  std::vector<PEID> offsets(num_elements);
  std::mt19937 gen(rank * 2 + 1);
  std::uniform_int_distribution<std::size_t> offset_distribution(
      0, std::numeric_limits<int32_t>::max());
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
  const auto& twolevel_comm = mpi::get_twolevel_columnmajor_communicators();
  get_timer().start_phase_measurement("partition_samples");
  const std::size_t num_needed_splitters =
      16 * std::log2(twolevel_comm.num_cols());
  const std::size_t num_local_samples = std::max(num_needed_splitters, 10ul);
  auto samples = sorting_internal::get_samples(elements, num_local_samples);
  get_timer().stop_phase_measurement("partition_samples");
  // get_timer().start_phase_measurement("partition_splitters");
  auto splitters =
      sorting_internal::select_splitters_level_1(twolevel_comm, samples, comp);
  // SEQ_EX(world_ctx, PRINT_VECTOR(splitters););
  //  get_timer().stop_phase_measurement("partition_splitters");
  const auto offsets = pseudo_random_offsets(elements.size(), world_ctx.rank());
  auto filter = False_Predicate{};
  auto transformer = [&](const T& t, const std::size_t&) { return t; };
  auto dst_computer = [&](const T& t, const std::size_t& i) {
    const auto it =
        std::upper_bound(splitters.begin(), splitters.end(), t, comp);
    const std::size_t offset_entry = i % offsets.size();
    const std::size_t colum_idx = std::distance(splitters.begin(), it);
    const std::size_t min_org_rank_column =
        twolevel_comm.min_org_rank_of_column(colum_idx);
    const std::size_t col_size = twolevel_comm.size_of_col_with(colum_idx);
    return min_org_rank_column + (offsets[offset_entry] % col_size);
  };
  auto partitioned_data = mpi::alltoall_combined(std::move(elements), filter,
                                                 transformer, dst_computer);
  constexpr bool debug = false;
  if constexpr (debug) {
    auto elems_per_group = allreduce_sum(partitioned_data.buffer.size(),
                                         twolevel_comm.get_col_ctx());
    SEQ_EX(
        world_ctx, if (twolevel_comm.get_col_ctx().rank() == 0) {
          PRINT_VAR(elems_per_group);
        });
  }
  elements = std::move(partitioned_data.buffer);
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
void secondlevel_partition(Container& elements, Comp comp = Comp{}) {
  using T = typename Container::value_type;
  const auto& twolevel_comm = mpi::get_twolevel_columnmajor_communicators();

  auto samples = get_samples(elements, twolevel_comm.get_col_ctx());
  auto splitters = select_splitters(samples, comp, twolevel_comm.get_col_ctx());
  auto filter = False_Predicate{};
  auto transformer = [&](const T& t, const std::size_t&) { return t; };
  auto dst_computer = [&](const T& t, const std::size_t&) {
    const auto it =
        std::upper_bound(splitters.begin(), splitters.end(), t, comp);
    return std::distance(splitters.begin(), it);
  };
  auto partitioned_data = mpi::twopass_alltoallv_openmp_special(
      std::move(elements), filter, transformer, dst_computer,
      twolevel_comm.get_col_ctx());
  elements = std::move(partitioned_data.buffer);
}
} // namespace sorting_internal

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
void twolevel_partition(Container& elements, Comp comp = Comp{}) {
  const bool debug = false;
  if constexpr (debug) {
    mpi::MPIContext ctx;
    const auto& twolevel_comm = mpi::get_twolevel_columnmajor_communicators();
    SEQ_EX(ctx, std::cout << "(" << twolevel_comm.row_index() << ", "
                          << twolevel_comm.col_index() << std::endl;);
  }

  sorting_internal::firstlevel_partition(elements, comp);
  sorting_internal::secondlevel_partition(elements, comp);
}

template <typename Container,
          typename Comp = std::less<typename Container::value_type>>
auto twolevel_sorting(Container& elements, Comp comp = Comp{}) {
  twolevel_partition(elements, comp);
  ips4o::parallel::sort(elements.begin(), elements.end(), comp);
  return elements;
}

} // namespace hybridMST
