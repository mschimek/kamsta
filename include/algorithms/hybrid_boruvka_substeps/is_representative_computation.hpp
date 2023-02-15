#pragma once

#include <tbb/parallel_for.h>

#include "definitions.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/context.hpp"
#include "util/utils.hpp"

namespace hybridMST {

class Representatives {
public:
  Representatives(std::size_t n) : predecessors(n), home_of_predecessors(n) {
    assign_initialize(predecessors, [](std::size_t&) {
      return reset_TMSB(VID_UNDEFINED);
    }); // most significant must not be set since this is used as a distinction
        // criterion later on.
    assign_initialize(home_of_predecessors, [](std::size_t&) { return -1; });
  }
  void reset_marks() {
    parallel_for(0, predecessors.size(), [&](std::size_t i) {
      predecessors[i] = reset_TMSB(predecessors[i]);
    });
  }
  auto compute_vertices_with_nonfinal_representatives() const {
    auto idx_to_query = parlay::sequence<std::uint8_t>::from_function(
        predecessors.size(),
        [&](const std::size_t& i) { return !is_MSB_set(predecessors[i]); });
    return parlay::pack_index(idx_to_query);
    // return parlay::filter(predecessors, [&](const VId& v) { return
    // !is_MSB_set(v);});
  }
  bool is_representative_final(std::size_t i) const {
    return is_MSB_set(predecessors[i]);
  }
  bool is_root(std::size_t i) const { return is_TMSB_set(predecessors[i]); }
  void set_final_representative(std::size_t i, VId representative) {
    predecessors[i] = set_MSB(representative);
  }
  void mark_as_root(std::size_t i, VId representative) {
    predecessors[i] = set_TMSB(representative);
  }
  void set_nonfinal_representative(std::size_t i, VId representative,
                                   PEID pe_representative) {
    predecessors[i] = representative;
    home_of_predecessors[i] = pe_representative;
  }
  non_init_vector<VId>& get_predecessors() { return predecessors; }
  non_init_vector<PEID>& get_home_of_predecessors() {
    return home_of_predecessors;
  }
  non_init_vector<VId> extract_predecessors() {
    return std::move(predecessors);
  }

private:
  non_init_vector<VId> predecessors;
  non_init_vector<PEID> home_of_predecessors;
};

struct IsRepresentative_Push {
  template <typename ET1, typename ET2>
  static bool is_src_part_of_pseudo_root(const ET1& local_edge,
                                         const ET2& remote_edge) {
    if (local_edge.get_dst() != remote_edge.get_src())
      return false;
    return true;
  }
  template <typename ET1, typename ET2>
  static bool is_src_root(const ET1& local_edge, const ET2& remote_edge) {
    if (local_edge.get_dst() != remote_edge.get_src())
      return false;
    const bool is_sum_even =
        (local_edge.get_src() + local_edge.get_dst()) % 2 == 0;
    return is_sum_even == (local_edge.get_src() < local_edge.get_dst());
  }

  template <typename Graph>
  static Representatives
  compute_representatives_(const non_init_vector<LocalEdgeId>& min_edge_idx,
                           const Graph& graph) {
    mpi::MPIContext ctx;
    const auto locator = graph.locator();
    // std::cout << locator << std::endl;
    const std::size_t n = graph.local_n();
    Representatives reps(n);

    if (n > 0) {
      if (locator.is_v_min_split()) {
        const VId local_id = graph.get_local_id(locator.v_min());
        reps.mark_as_root(local_id, locator.v_min());
      }
      if (locator.is_v_max_split()) {
        const VId local_id = graph.get_local_id(locator.v_max());
        reps.mark_as_root(local_id, locator.v_max());
      }
    }
    auto filter = [&](const LocalEdgeId& elem, const std::size_t idx) -> bool {
      if (!is_defined(elem))
        reps.set_final_representative(idx, reset_TMSB(VID_UNDEFINED));
      if (reps.is_representative_final(idx))
        return true;
      else {
        const auto& edge = graph.edges()[elem];
        const Edge flipped_edge{edge.get_dst(), edge.get_src()};
        const auto [dst_min_pe, is_dst_split] =
            locator.get_min_pe_split_info(flipped_edge);
        if (is_dst_split) {
          reps.set_final_representative(idx, edge.get_dst());
        } else {
          reps.set_nonfinal_representative(idx, edge.get_dst(), dst_min_pe);
        }
      }
      return static_cast<bool>(reps.is_representative_final(idx));
    };
    auto transformer = [&](const LocalEdgeId& elem, const std::size_t&) {
      const auto& edge = graph.edges()[elem];
      return Edge{edge.get_src(), edge.get_dst()};
    };
    auto dst_computer = [&](const LocalEdgeId& elem, const size_t& /*idx*/) {
      const auto& edge = graph.edges()[elem];
      const Edge flipped_edge{edge.get_dst(), edge.get_src()};
      const auto pe = locator.get_min_pe(flipped_edge);
      return pe;
    };
    auto recv =
        mpi::alltoall_combined(min_edge_idx, filter, transformer, dst_computer);

    parallel_for(0, recv.buffer.size(), [&](std::size_t i) {
      const auto& remote_edge = recv.buffer[i];
      const VId local_id = graph.get_local_id(remote_edge.get_dst());
      const VId min_edge_id = min_edge_idx[local_id];
      const auto& local_edge = graph.edges()[min_edge_id];
      if (!is_src_part_of_pseudo_root(local_edge, remote_edge)) {
        return;
      }
      if (is_src_root(local_edge, remote_edge)) {
        reps.mark_as_root(local_id, local_edge.get_src());
      } else {
        reps.set_final_representative(local_id, local_edge.get_dst());
      }
    });
    return reps;
  }
};
} // namespace hybridMST
