#pragma once

#include "datastructures/growt.hpp"

#include "definitions.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/context.hpp"
#include "mpi/scan.hpp"
#include "util/utils.hpp"

namespace hybridMST {

struct ExchangeRepresentativesPush_Sort {
  using TaskBeginEnd = std::pair<std::size_t, std::size_t>;
  struct NamePrevName {
    VId name;
    VId prev_name;
    friend std::ostream& operator<<(std::ostream& out,
                                    const NamePrevName& elem) {
      return out << "(name: " << elem.name << ", prev: " << elem.prev_name
                 << ")";
    }
  };

  /// dst_pe[i] contains the destination pe for the information about the new
  /// src representative for the i-th edge or -1 if it is a duplicate or dst
  /// belongs to the same rank.
  /// Note that it is possible that the back edge to an input edge e does not
  /// exists if the input edge e = (u,v) is not the lightest edge among all
  /// edges from u to v and the filtering process has not discarded e (as the
  /// other (u,v) are located on another PE)
  template <typename Edges, typename Locator>
  static void process_batch_sorting(TaskBeginEnd task,
                                    non_init_vector<PEID>& dst_pe,
                                    const Edges& edges,
                                    const Locator& locator) {
    mpi::MPIContext ctx;
    const auto& [begin, end] = task;
    const bool is_task_empty = begin == end;
    if (is_task_empty) return;
    VId prev_v = edges[begin].get_src();
    const PEID sentinel = -1;
    PEID prev_pe = sentinel;

    for (std::size_t i = begin; i < end; ++i) {
      const auto& cur_edge = edges[i];
      if (prev_v != cur_edge.get_src()) {
        prev_pe = sentinel;  // reset pe
      }
      const Edge back_edge{cur_edge.get_dst(), cur_edge.get_src()};
      const PEID pe = locator.get_min_pe_or_sentinel(back_edge, sentinel);
      const bool is_duplicate = prev_pe == pe || pe == ctx.rank();
      if (is_duplicate)
        dst_pe[i] = sentinel;
      else
        dst_pe[i] = pe;
      prev_pe = pe;
      prev_v = cur_edge.get_src();
    }
  }

  [[nodiscard]] static TaskBeginEnd compute_task_boundaries(
      std::size_t size, std::size_t num_tasks, std::size_t task_id) {
    const std::size_t task_size = size / num_tasks;
    const std::size_t remainder = size % num_tasks;
    const bool is_last_task = num_tasks == 1 + task_id;
    const std::size_t begin_idx = task_id * task_size;
    const std::size_t end_idx =
        (task_id + 1) * task_size + (is_last_task ? remainder : 0);
    return std::make_pair(begin_idx, end_idx);
  }

  // assumptions:
  // - edges are sorted with respect to (src, dst)
  // - there are not duplicate edges splitted between processor (since only one
  // of them would get the update)
  template <typename Graph, typename Container>
  static auto execute(const Graph& graph,
                      const Container& new_reps_local_vertices,
                      std::size_t round) {
    using EdgeType = typename Graph::EdgeType;
    mpi::MPIContext ctx;
    const auto& locator = graph.locator();

    // assert: edges are sorted with respect to SrcDst

    // batch of vertices for each threads:
    get_timer().start_phase("label_exchange");

    get_timer().start("exchange_representatives_filter", round);
    const std::size_t num_tasks = ctx.threads_per_mpi_process();
    non_init_vector<PEID> dst_pe(graph.edges().size());

    parallel_for(0, num_tasks, [&](std::size_t i) {
      const TaskBeginEnd task =
          compute_task_boundaries(graph.edges().size(), num_tasks, i);
      process_batch_sorting(task, dst_pe, graph.edges(), locator);
    });
    get_timer().stop("exchange_representatives_filter", round);

    get_timer().start("exchange_representatives_exchange", round);
    const PEID sentinel_pe = -1;
    auto filter = [&](const EdgeType&, const std::size_t& i) {
      return dst_pe[i] == sentinel_pe;
    };
    auto transformer = [&](const EdgeType& edge, const std::size_t&) {
      const VId src = edge.get_src();
      const VId local_id = graph.get_local_id(src);
      return NamePrevName{new_reps_local_vertices[local_id], src};
    };
    auto dst_computer = [&](const EdgeType&, const std::size_t& i) {
      return dst_pe[i];
    };
    auto rep_info = mpi::alltoall_combined(graph.edges(), filter, transformer,
                                           dst_computer);
    get_timer().stop("exchange_representatives_exchange", round);

    get_timer().start("exchange_representatives_allocate_map", round);
    REORDERING_BARRIER
    const std::size_t map_size = rep_info.buffer.size() * 1.1;
    growt::GlobalVIdMap<VId> grow_map{map_size};
    REORDERING_BARRIER
    get_timer().stop("exchange_representatives_allocate_map", round);

    get_timer().start("exchange_representatives_write_map", round);
    REORDERING_BARRIER {
      parallel_for(0, rep_info.buffer.size(), [&](std::size_t i) {
        const auto& elem = rep_info.buffer[i];
        static_assert(sizeof(VId) >= 8);
        grow_map.insert(elem.prev_name + 1, elem.name);
      });
    }
    REORDERING_BARRIER
    get_timer().stop("exchange_representatives_write_map", round);
    get_timer().stop_phase();
    return grow_map;
  }
};
struct ExchangeRepresentativesPush_Hash {
  template <typename Buffer>
  static auto foo(const Buffer& rep_info) {
    mpi::MPIContext ctx;
    growt::GlobalVIdMap<VId> grow_map{rep_info.buffer.size()};
    {
      parallel_for(0, rep_info.buffer.size(), [&](std::size_t i) {
        auto& elem = rep_info.buffer[i];
        static_assert(sizeof(VId) >= 8);
        auto [it, _] = grow_map.insert(elem.prev_name + 1, elem.name);
        if (it == grow_map.end())
          std::cout << ctx.rank() << "--" << elem << std::endl;
      });
    }
    return grow_map;
  }
  struct NamePrevName {
    VId name;
    VId prev_name;
    friend std::ostream& operator<<(std::ostream& out,
                                    const NamePrevName& elem) {
      return out << "(name: " << elem.name << ", prev: " << elem.prev_name
                 << ")";
    }
  };
  static VId combine(PEID rank, VId v) {
    VId combined = static_cast<VId>(rank) << 44;
    return combined | v;
  }
  static std::pair<PEID, VId> split(VId combined_entry) {
    PEID pe = static_cast<PEID>(combined_entry >> 44);
    constexpr VId ones = ~VId{0};
    constexpr VId mask_44 = ones >> 20;
    return std::make_pair(static_cast<PEID>(pe), combined_entry & mask_44);
  }

  template <typename Edges, typename Locator>
  static auto filter(const Edges& edges, const Locator& locator) {
    mpi::MPIContext ctx;
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        edges.size(), parlay::hash_numeric<VId>{});

    parallel_for(0, edges.size(), [&](std::size_t i) {
      const auto& cur_edge = edges[i];
      const PEID pe =
          locator.get_min_pe(Edge{cur_edge.get_dst(), cur_edge.get_src()});
      if (pe == ctx.rank()) {
        return;
      }
      const VId combined_entry = combine(pe, cur_edge.get_src());
      table.insert(combined_entry);
    });
    return table.entries();
  }
  // assumptions:
  // - there are not duplicate edges splitted between processor (since only one
  // of them would get the update)
  template <typename Graph>
  static auto execute(const Graph& graph,
                      const non_init_vector<VId>& new_reps_local_vertices,
                      std::size_t round) {
    mpi::MPIContext ctx;
    get_timer().start("exchange_representatives_filter", round);
    const auto peid_v_entries = filter(graph.edges(), graph.locator());
    get_timer().stop("exchange_representatives_filter", round);
    get_timer().start("exchange_representatives_exchange", round);
    auto filter = False_Predicate{};
    auto transformer = [&](const VId& combined_entry, const std::size_t&) {
      const auto& [pe, v] = split(combined_entry);
      const VId local_id = graph.get_local_id(v);
      return NamePrevName{new_reps_local_vertices[local_id], v};
    };
    auto dst_computer = [&](const VId& combined_entry, const std::size_t&) {
      const auto& [pe, v] = split(combined_entry);
      return pe;
    };
    auto rep_info = mpi::alltoall_combined(peid_v_entries, filter, transformer,
                                           dst_computer);
    get_timer().stop("exchange_representatives_exchange", round);
    get_timer().start("exchange_representatives_write_map", round);

    auto grow_map = foo(rep_info);

    get_timer().stop("exchange_representatives_write_map", round);
    return grow_map;
  }
};

}  // namespace hybridMST
