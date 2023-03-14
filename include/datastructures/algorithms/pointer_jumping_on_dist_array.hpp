#pragma once

#include <algorithm>

#include "datastructures/distributed_array.hpp"
#include "definitions.hpp"
#include "mpi/twolevel_alltoall.hpp"
#include "shared_mem_parallel.hpp"

namespace hybridMST {

template <typename T>
struct ParallelPointerJumping {

  static Span<std::size_t> keep_non_root_elemens(
      const DistributedArray<T>& array,
      Span<std::size_t> global_idx_to_jump_from) {
    auto non_root_entries =
        parlay::filter(global_idx_to_jump_from, [&](std::size_t v_global) {
          const T value = array.get_value_locally(v_global);
          return !is_MSB_set(value);
        });
    parallel_for(0, non_root_entries.size(), [&](std::size_t i) {
      global_idx_to_jump_from[i] = non_root_entries[i];
    });
    return Span(global_idx_to_jump_from.data(), non_root_entries.size());
  }
  static non_init_vector<std::size_t> get_global_idxs_to_jump_from(
      DistributedArray<T>& array) {
    using CachelineAlignedInt = CachelineAlignedType<std::size_t>;
    mpi::MPIContext ctx;
    const std::size_t num_threads = ctx.threads_per_mpi_process();
    non_init_vector<CachelineAlignedInt> counts(num_threads, 0ull);
    const auto is_value_at_index_root = [&](std::size_t i) {
      const T& value = array.get_value_locally(i);
      return is_MSB_set(value);
    };
    non_init_vector<VId> global_idxs_to_jump_from;
#pragma omp parallel
    {
      const auto thread_id = omp_get_thread_num();
#pragma omp for schedule(static)
      for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
        if (!is_value_at_index_root(i)) {
          ++counts[thread_id];
        }
      }
#pragma omp single
      {
        std::inclusive_scan(counts.begin(), counts.end(), counts.begin());
        global_idxs_to_jump_from.resize(counts.back());
      }

#pragma omp for schedule(static)
      for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
        if (!is_value_at_index_root(i)) {
          --counts[thread_id];
          global_idxs_to_jump_from[counts[thread_id]] = i;
        }
      }
    }
    return global_idxs_to_jump_from;
  }

  static Span<std::size_t> execute_one_jump(
      DistributedArray<T>& array, Span<std::size_t> global_idx_to_jump_from) {
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
      const IndexValue& idx_value = elem.payload();
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
    parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
      const auto& index_value = reply.buffer[i];
      array.set_value_locally(index_value);
    });
    return keep_non_root_elemens(array, global_idx_to_jump_from);
  }

  static Span<std::size_t> execute_one_jump_filter(
      DistributedArray<T>& array, Span<std::size_t> global_idx_to_jump_from) {
    using IndexValue = typename DistributedArray<T>::IndexValue;
    mpi::MPIContext ctx;
    // auto global_v_parents = filter_out_duplicates(global_v_to_jump_from,
    // [&](const VId& v_global) { return parents.get_parent(v_global); });
    const std::size_t num_queries = global_idx_to_jump_from.size();
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        num_queries, parlay::hash_numeric<VId>{});

    parallel_for(0, num_queries, [&](std::size_t i) {
      const VId global_id_to_query = global_idx_to_jump_from[i];
      const VId predecessor = array.get_value_locally(global_id_to_query);
      table.insert(predecessor);
    });
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
      const std::size_t& value = elem.payload();
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
    const std::size_t map_size = reply.buffer.size() * 1.2;
    growt::GlobalVIdMap<VId> grow_map{map_size};
    parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
      const auto [requested_predecessor, replied_predecessor] = reply.buffer[i];
      const auto [it, _] =
          grow_map.insert(requested_predecessor + 1, replied_predecessor);
      if (it == grow_map.end()) {
        std::cout << "growt wrong insert" << std::endl;
        std::abort();
      }
    });
    parallel_for(0, global_idx_to_jump_from.size(), [&](std::size_t i) {
      const VId local_id_to_query = global_idx_to_jump_from[i];
      const VId& predecessor = array.get_value_locally(local_id_to_query);
      if (is_MSB_set(predecessor)) {
        return;
      }
      auto it = grow_map.find(predecessor + 1);
      array.set_value_locally(local_id_to_query, (*it).second);
    });
    return keep_non_root_elemens(array, global_idx_to_jump_from);
  }
  template <typename IsRoot>
  static void execute(DistributedArray<T>& array, IsRoot& is_root) {
    mpi::MPIContext ctx;
    static_assert(std::is_integral_v<T>);  // values will be used as indices
                                           // during jumping
    static_assert(std::is_unsigned_v<T>);
    parallel_for(array.index_begin(), array.index_end(), [&](std::size_t i) {
      auto& elem = array.get_value_locally(i);
      if (is_root(i, elem)) elem = set_MSB(elem);
    });
    auto global_idxs_to_jump_from = get_global_idxs_to_jump_from(array);
    Span<std::size_t> global_idxs_to_jump_from_span{global_idxs_to_jump_from};
    do {
      global_idxs_to_jump_from_span =
          execute_one_jump_filter(array, global_idxs_to_jump_from_span);
    } while (mpi::allreduce_max(global_idxs_to_jump_from_span.size()) > 0ull);
    parallel_for(array.index_begin(), array.index_end(), [&](std::size_t i) {
      auto& elem = array.get_value_locally(i);
      elem = reset_MSB(elem);
    });
  }
};
}  // namespace hybridMST
