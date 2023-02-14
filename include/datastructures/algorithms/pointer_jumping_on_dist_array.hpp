#pragma once

#include <algorithm>
#include <execution>

#include "datastructures/distributed_array.hpp"
#include "definitions.hpp"
#include "mpi/twolevel_alltoall.hpp"

namespace hybridMST {

template <typename T> struct ParallelPointerJumping {

  static non_init_vector<std::size_t>
  get_global_idxs_to_jump_from(DistributedArray<T>& array) {
    using CachelineAlignedInt = CachelineAlignedType<std::size_t>;
    mpi::MPIContext ctx;
    const std::size_t num_threads = ctx.threads_per_mpi_process();
    non_init_vector<CachelineAlignedInt> counts(num_threads, 0ull);
    const auto is_value_at_index_root = [&](std::size_t i) {
      const T& value = array.get_value_locally(i);
      return is_MSB_set(value);
    };
#pragma omp parallel for schedule(static)
    for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
      const auto thread_id = omp_get_thread_num();
      if (is_value_at_index_root(i)) {
        continue;
      }
      ++counts[thread_id];
    }
    std::inclusive_scan(counts.begin(), counts.end(), counts.begin());
    non_init_vector<VId> global_idxs_to_jump_from(counts.back());

#pragma omp parallel for schedule(static)
    for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
      const auto thread_id = omp_get_thread_num();
      if (is_value_at_index_root(i)) {
        continue;
      }
      --counts[thread_id];
      global_idxs_to_jump_from[counts[thread_id]] = i;
    }
    return global_idxs_to_jump_from;
  }

  static Span<std::size_t>
  execute_one_jump(DistributedArray<T>& array,
                   Span<std::size_t> global_idx_to_jump_from) {
    using IndexValue = typename DistributedArray<T>::IndexValue;
    mpi::MPIContext ctx;
    // auto global_v_parents = filter_out_duplicates(global_v_to_jump_from,
    // [&](const VId& v_global) { return parents.get_parent(v_global); });

    auto filter = False_Predicate{};
    auto transformer = [&](const std::size_t& i, const std::size_t) {
      return IndexValue{i, array.get_value_locally(i)};
    };
    auto dst_computer = [&](const std::size_t& i, const std::size_t) {
      const T& value = array.get_value_locally(i);
      return array.get_pe(value);
    };

    auto request = mpi::two_level_alltoall(global_idx_to_jump_from, filter,
                                           transformer, dst_computer);

    auto filter_reply = False_Predicate{};
    auto transformer_reply = [&](const auto& elem, const std::size_t) {
      const IndexValue& idx_value = elem.payload;
      return IndexValue{idx_value.index,
                        array.get_value_locally(idx_value.value)};
    };
    // auto transformer_reply = [&](const IndexValue& idx_value,
    //                              const std::size_t) {
    //   return IndexValue{idx_value.index,
    //                     array.get_value_locally(idx_value.value)};
    // };
    // auto dst_computer_reply = [&](const IndexValue&, const std::size_t i) {
    //   return request.get_pe(i);
    // };
    auto return_sender = [](const auto& elem, std::size_t) {
      return elem.get_sender();
    };
    auto reply = mpi::two_level_alltoall_extract(
        request.buffer, filter_reply, transformer_reply, return_sender);
    MPI_ASSERT_(are_keys_unique(reply.buffer,
                                [](const IndexValue& elem) { return elem; }),
                "vertex_parents are not unique");
#pragma omp parallel for
    for (std::size_t i = 0; i < reply.buffer.size(); ++i) {
      const auto& index_value = reply.buffer[i];
      array.set_value_locally(index_value);
    }
    auto it = std::remove_if(
        std::execution::par, global_idx_to_jump_from.begin(),
        global_idx_to_jump_from.end(), [&](const std::size_t& v_global) {
          const T value = array.get_value_locally(v_global);
          return is_MSB_set(value);
        });
    const std::size_t num_remainings_non_root_entries =
        static_cast<std::size_t>(
            std::distance(global_idx_to_jump_from.begin(), it));
    return Span(global_idx_to_jump_from.data(),
                num_remainings_non_root_entries);
  }

  static Span<std::size_t>
  execute_one_jump_filter(DistributedArray<T>& array,
                          Span<std::size_t> global_idx_to_jump_from) {
    using IndexValue = typename DistributedArray<T>::IndexValue;
    mpi::MPIContext ctx;
    // auto global_v_parents = filter_out_duplicates(global_v_to_jump_from,
    // [&](const VId& v_global) { return parents.get_parent(v_global); });
    const std::size_t num_queries = global_idx_to_jump_from.size();
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        num_queries, parlay::hash_numeric<VId>{});

#pragma omp parallel for
    for (std::size_t i = 0; i < num_queries; ++i) {
      const VId global_id_to_query = global_idx_to_jump_from[i];
      const VId predecessor = array.get_value_locally(global_id_to_query);
      table.insert(predecessor);
    }
    auto unique_predecessors = table.entries();
    auto filter = False_Predicate{};
    auto transformer = [&](const std::size_t& value, const std::size_t) {
      return value;
    };
    auto dst_computer = [&](const std::size_t& value, const std::size_t) {
      return array.get_pe(value);
    };

    auto request = mpi::two_level_alltoall(unique_predecessors, filter,
                                           transformer, dst_computer);

    auto filter_reply = False_Predicate{};
    auto transformer_reply = [&](const auto& elem, const std::size_t) {
      const std::size_t& value = elem.payload;
      return IndexValue{value, array.get_value_locally(value)};
    };
    // auto transformer_reply = [&](const IndexValue& idx_value,
    //                              const std::size_t) {
    //   return IndexValue{idx_value.index,
    //                     array.get_value_locally(idx_value.value)};
    // };
    // auto dst_computer_reply = [&](const IndexValue&, const std::size_t i) {
    //   return request.get_pe(i);
    // };
    auto return_sender = [](const auto& elem, std::size_t) {
      return elem.get_sender();
    };
    auto reply = mpi::two_level_alltoall_extract(
        request.buffer, filter_reply, transformer_reply, return_sender);
    // MPI_ASSERT_(are_keys_unique(reply.buffer,
    //                             [](const IndexValue& elem) { return elem; }),
    //             "vertex_parents are not unique");
    growt::GlobalVIdMap<VId> grow_map{reply.buffer.size() * 1.2};
#pragma omp parallel for
    for (std::size_t i = 0; i < reply.buffer.size(); ++i) {
      const auto [requested_predecessor, replied_predecessor] = reply.buffer[i];
      const auto [it, _] =
          grow_map.insert(requested_predecessor + 1, replied_predecessor);
      if (it == grow_map.end()) {
        std::cout << "growt wrong insert" << std::endl;
        std::abort();
      }
    }
#pragma omp parallel for
    for (std::size_t i = 0; i < global_idx_to_jump_from.size(); ++i) {
      const VId local_id_to_query = global_idx_to_jump_from[i];
      const VId& predecessor = array.get_value_locally(local_id_to_query);
      if (is_MSB_set(predecessor))
        continue;
      auto it = grow_map.find(predecessor + 1);
      array.set_value_locally(local_id_to_query, (*it).second);
    }
    auto it = std::remove_if(
        std::execution::par, global_idx_to_jump_from.begin(),
        global_idx_to_jump_from.end(), [&](const std::size_t& v_global) {
          const T value = array.get_value_locally(v_global);
          return is_MSB_set(value);
        });
    const std::size_t num_remainings_non_root_entries =
        static_cast<std::size_t>(
            std::distance(global_idx_to_jump_from.begin(), it));
    return Span(global_idx_to_jump_from.data(),
                num_remainings_non_root_entries);
  }
  template <typename IsRoot>
  static void execute(DistributedArray<T>& array, IsRoot& is_root) {
    mpi::MPIContext ctx;
    static_assert(
        std::is_integral_v<T>); // values will be used as indices during jumping
    static_assert(std::is_unsigned_v<T>);
#pragma omp parallel for
    for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
      auto& elem = array.get_value_locally(i);
      if (is_root(i, elem))
        elem = set_MSB(elem);
    }
    auto global_idxs_to_jump_from = get_global_idxs_to_jump_from(array);
    Span<std::size_t> global_idxs_to_jump_from_span{global_idxs_to_jump_from};
    do {
      global_idxs_to_jump_from_span =
          execute_one_jump_filter(array, global_idxs_to_jump_from_span);
    } while (mpi::allreduce_max(global_idxs_to_jump_from_span.size()) > 0ull);
#pragma omp parallel for
    for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
      auto& elem = array.get_value_locally(i);
      elem = reset_MSB(elem);
    }
  }
};
} // namespace hybridMST
