#pragma once

#include <mpi.h>

#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mpi/twolevel_alltoall_helpers.hpp"

namespace hybridMST::mpi {

namespace two_level_alltoall_impl {
template <typename Tasks>
void compute_bucket_counts(Tasks& tasks, std::size_t num_recv,
                           std::vector<std::size_t>& send_counts,
                           std::vector<std::size_t>& send_displs) {
  send_counts = std::vector<std::size_t>(num_recv, 0ull);
  send_displs = std::vector<std::size_t>(num_recv, 0ull);
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    for (std::size_t j = 0; j < num_recv; ++j) {
      const auto v = send_counts[j];
      send_counts[j] += tasks[i].send_counts[j];
      tasks[i].send_counts[j] = v;
    }
  }
  std::exclusive_scan(send_counts.begin(), send_counts.end(),
                      send_displs.begin(), 0ull);
}

namespace first_hop {
template <typename Container, typename Filter, typename DstCalculator,
          typename Tasks>
non_init_vector<PEID> first_pass(Tasks& tasks, const Container& data,
                                 Filter&& filter,
                                 DstCalculator&& dstCalculator) {
  const auto& grid_communicators = get_grid_communicators();
  non_init_vector<PEID> final_destinations(data.size());
  parallel_for(0, tasks.size(), [&](std::size_t i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      if (filter(elem, j)) {
        final_destinations[j] = -1;
      } else {
        const PEID final_destination = dstCalculator(elem, j);
        const PEID intermediate_destination =
            grid_communicators.col_index(final_destination);

        final_destinations[j] = final_destination;
        ++task.send_counts[intermediate_destination];
      }
    }
  });
  return final_destinations;
}

template <typename SendMessages, typename Container, typename Transformer,
          typename Tasks>
void second_pass(SendMessages& send_messages, Tasks& tasks,
                 const Container& data,
                 const non_init_vector<PEID>& final_destinations,
                 const std::vector<std::size_t>& send_displs,
                 Transformer&& transformer) {
  using MessageType = typename SendMessages::MessageType;
  const auto& grid_communicators = get_grid_communicators();
  mpi::MPIContext ctx;
  parallel_for(0, tasks.size(), [&](std::size_t i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      const auto final_destination = final_destinations[j];
      if (final_destination != -1) {

        const PEID intermediate_destination =
            grid_communicators.col_index(final_destination);
        const auto idx = send_displs[intermediate_destination] +
                         task.send_counts[intermediate_destination];
        send_messages[idx] = MessageType{transformer(elem, j)};
        send_messages[idx].set_sender(ctx.rank());
        send_messages[idx].set_receiver(final_destination);
        ++task.send_counts[intermediate_destination];
      }
    }
  });
}

template <typename Container, typename Filter, typename Transformer,
          typename DstCalculator>
inline auto two_level_alltoall(Container&& data, Filter&& filter,
                               Transformer&& transformer,
                               DstCalculator&& dstCalculator) {
  using Container_ = std::decay_t<Container>;
  using TransformedType =
      std::invoke_result_t<Transformer, const typename Container_::value_type&,
                           const std::size_t&>;
  using MessageType = Message<TransformedType>;

  mpi::MPIContext ctx;
  const auto& grid_communicators = get_grid_communicators();
  const auto& row_ctx = grid_communicators.get_row_ctx();
  const std::size_t num_threads = ctx.threads_per_mpi_process();
  auto tasks = create_tasks(data.size(), num_threads, row_ctx.size());

  get_timer().start_phase_measurement("twopass_loop1");
  const auto final_destinations =
      first_pass(tasks, data, filter, dstCalculator);
  get_timer().stop_phase_measurement("twopass_loop1");

  get_timer().start_phase_measurement("twopass_accounting");
  // compute 2D-exclusive scan cache-friendly
  std::vector<std::size_t> send_counts;
  std::vector<std::size_t> send_displs;
  compute_bucket_counts(tasks, row_ctx.size(), send_counts, send_displs);
  PreallocMessages<MessageType> send_messages(send_counts);
  get_timer().stop_phase_measurement("twopass_accounting");

  get_timer().start_phase_measurement("twopass_loop2");
  second_pass(send_messages, tasks, data, final_destinations, send_displs,
              transformer);
  get_timer().stop_phase_measurement("twopass_loop2");
  if constexpr (std::is_rvalue_reference_v<Container&&>) {
    dump(data);
  }

  return alltoallv(send_messages, row_ctx);
}
} // namespace first_hop

namespace second_hop {
template <typename Tasks, typename Container>
void first_pass(Tasks& tasks, const Container& data) {

  const auto& grid_communicators = get_grid_communicators();
  parallel_for(0, tasks.size(), [&](std::size_t i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      const int& row_destination =
          grid_communicators.row_index(elem.get_receiver());
      ++task.send_counts[row_destination];
    }
  });
}

template <typename SendMessages, typename Container, typename Tasks,
          bool extract_payload>
void second_pass(SendMessages& send_messages, Tasks& tasks,
                 const Container& data,
                 const std::vector<std::size_t>& send_displs) {
  const auto& grid_communicators = get_grid_communicators();
  parallel_for(0, tasks.size(), [&](std::size_t i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      const PEID row_destination =
          grid_communicators.row_index(elem.get_receiver());
      const auto idx =
          send_displs[row_destination] + task.send_counts[row_destination];
      if constexpr (extract_payload) {
        send_messages[idx] = elem.payload();
      } else {
        send_messages[idx] = elem;
      }
      ++task.send_counts[row_destination];
    }
  });
}

template <typename Container, bool extract_payload = false>
inline auto two_level_alltoall(Container& data) {

  mpi::MPIContext ctx;
  const auto& col_ctx = get_grid_communicators().get_col_ctx();
  const std::size_t num_threads = ctx.threads_per_mpi_process();
  auto tasks = create_tasks(data.size(), num_threads, col_ctx.size());

  get_timer().start_phase_measurement("twopass_loop1");
  first_pass(tasks, data);
  get_timer().stop_phase_measurement("twopass_loop1");

  get_timer().start_phase_measurement("twopass_accounting");
  // compute 2D-exclusive scan cache-friendly
  std::vector<std::size_t> send_counts;
  std::vector<std::size_t> send_displs;
  compute_bucket_counts(tasks, col_ctx.size(), send_counts, send_displs);

  using Container_ = std::decay_t<Container>;
  using MessageType = typename Container_::value_type;
  using UnderlyingPayloadType = typename MessageType::Payload;
  using AdjustedMessageType =
      std::conditional_t<extract_payload, UnderlyingPayloadType, MessageType>;
  using SendMessages = PreallocMessages<AdjustedMessageType>;
  SendMessages send_messages(send_counts);
  get_timer().stop_phase_measurement("twopass_accounting");

  get_timer().start_phase_measurement("twopass_loop2");
  using Tasks = decltype(tasks);
  second_pass<SendMessages, Container, Tasks, extract_payload>(
      send_messages, tasks, data, send_displs);
  get_timer().stop_phase_measurement("twopass_loop2");

  dump(data); // can always be dumped as this is intermediate data in second hop
              // which will never be used later on
  return alltoallv(send_messages, col_ctx);
}
} // namespace second_hop
} // namespace two_level_alltoall_impl

///@brief MPI alltoallv exchange using a two-level approach.
///
/// The p MPI process (from MPI_COMM_WORLD) are ordered in a quadratic grid with
/// length sqrt(p) A message from process p to process q is delievered in two
/// steps:
/// Firstly,  a message from [row(p), column(p)] to [row(p), colum(q)] is
/// sent (row-wise alltoall).
/// Secondly, a message from [row(p), column(q)] to
/// [row(q), colum(q)] is sent (column-wise alltoall).
///
/// Note: In case we do not have a square number, e.g. 18 the processors are
/// ordered as follows:
/// 0  1  2  3
/// 4  5  6  7
/// 8  9  10 11
/// 12 13 14 15
/// 16 17
/// Since we cannot send a message from PE 16
/// to PE 2 (etc.) in the above-described scheme, we use the following virtual
/// topology for the first row-wise exchange:
/// 0  1  2  3 (16)
/// 4  5  6  7 (17)
/// 8  9  10 11
/// 12 13 14 15
template <bool extract_payload, typename Container, typename Filter,
          typename Transformer, typename DstCalculator>
auto two_level_alltoall_impll(Container&& data, Filter&& filter,
                              Transformer&& transformer,
                              DstCalculator&& dstCalculator) {

  mpi::MPIContext ctx;
  auto intermediate_msgs =
      two_level_alltoall_impl::first_hop::two_level_alltoall(
          std::forward<Container>(data), std::forward<Filter>(filter),
          std::forward<Transformer>(transformer),
          std::forward<DstCalculator>(dstCalculator));
  return two_level_alltoall_impl::second_hop::two_level_alltoall<
      decltype(intermediate_msgs.buffer), extract_payload>(
      intermediate_msgs.buffer);
}

template <typename Container, typename Filter, typename Transformer,
          typename DstCalculator>
auto two_level_alltoall(Container&& data, Filter&& filter,
                        Transformer&& transformer,
                        DstCalculator&& dstCalculator) {

  constexpr bool extract_payload = false;
  return two_level_alltoall_impll<extract_payload>(
      std::forward<Container>(data), std::forward<Filter>(filter),
      std::forward<Transformer>(transformer),
      std::forward<DstCalculator>(dstCalculator));
}

template <typename Container, typename Filter, typename Transformer,
          typename DstCalculator>
auto two_level_alltoall_extract(Container&& data, Filter&& filter,
                                Transformer&& transformer,
                                DstCalculator&& dstCalculator) {
  constexpr bool extract_payload = true;
  return two_level_alltoall_impll<extract_payload>(
      std::forward<Container>(data), std::forward<Filter>(filter),
      std::forward<Transformer>(transformer),
      std::forward<DstCalculator>(dstCalculator));
}

} // namespace hybridMST::mpi
