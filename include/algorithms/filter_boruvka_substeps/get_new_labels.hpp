#pragma once

#include <vector>

#include "parlay/hash_table.h"
#include "parlay/primitives.h"

#include "algorithms/hybrid_boruvka_substeps/edge_renaming.hpp"
#include "algorithms/hybrid_boruvka_substeps/get_ghost_representatives.hpp"
#include "datastructures/distributed_graph.hpp"
#include "datastructures/distributed_parent_array.hpp"
#include "definitions.hpp"
#include "util/timer.hpp"

namespace hybridMST {

struct RelabelViaPush {

  template <typename T, typename... Args>
  static auto get_max_num_local_vertices(const std::vector<T, Args...>& edges) {
    if (edges.empty()) {
      return 0ull;
    }
    const auto [min_it, max_it] = parlay::minmax_element(edges, SrcOrder<T>{});
    return (max_it->get_src() - min_it->get_src()) + 1ull;
  }
  /// the phase concurrent hash table has a very poor performance if the table
  /// size is "too" small
  static std::size_t get_table_size(const std::size_t num_elements) {
    std::size_t min_size = 5'000;
    return std::max(min_size,
                    static_cast<std::size_t>((num_elements + 10) * 1.5));
  }
  template <typename Container, typename Graph>
  static non_init_vector<VId>
  transform_from_list_to_vector(const Container& v_parent, const Graph& graph) {
    non_init_vector<VId> parents(graph.local_n());
    if (graph.local_n() != v_parent.size()) {
      PRINT_VAR(graph.local_n());
      PRINT_VAR(v_parent.size());
    }
    parallel_for(0, v_parent.size(), [&](const std::size_t& i) {
      const auto [v, parent] = v_parent[i];
      parents[graph.get_local_id(v)] = parent;
    });
    return parents;
  }

  template <typename T, typename... Args>
  static auto get_src_vertices(const std::vector<T, Args...>& edges,
                               int threshold_weight, int round) {
    const std::size_t num_local_vertices = get_max_num_local_vertices(edges);
    const std::size_t table_size = get_table_size(num_local_vertices);
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        table_size, parlay::hash_numeric<VId>{});
    parallel_for(0, edges.size(), [&](const std::size_t& i) {
      auto& edge = edges[i];
      if (edge.get_weight() <= threshold_weight) {
      std::cout << "should not be called " << std::endl;
        return;
      }
      table.insert(edge.get_src());
    });
    auto entries = table.entries();
    return entries;
  }
  template <typename Container>
  static void execute(int round, Container& edges,
                      const ParentArray& parent_array,
                      Weight threshold_weight) {

    get_timer().start("via_push_relabel_preprocess", round);
    mpi::MPIContext ctx;
    using EdgeType = typename Container::value_type;
    constexpr bool compactify_graph = true;
    const auto comp = SrcDstWeightOrder<EdgeType>{};
    if (!edges.empty() &&
        !parlay::is_sorted(edges,
                           comp)) { // parlay's is_sorted crashes with empty range
      ips4o::parallel::sort(edges.begin(), edges.end(), comp);
    }
    DistributedGraph<EdgeType, compactify_graph> graph(edges, round);
    get_timer().stop("via_push_relabel_preprocess", round);
    get_timer().start("via_push_relabel_src_vertices", round);
    const auto src_vertices = get_src_vertices(edges, threshold_weight, round);
    get_timer().stop("via_push_relabel_src_vertices", round);
    get_timer().start("via_push_relabel_get_parents", round);
    const auto v_parents =
        parent_array.get_parents(src_vertices, ParentArray::InVector{});
    const auto parents = transform_from_list_to_vector(v_parents, graph);
    get_timer().stop("via_push_relabel_get_parents", round);
    get_timer().start("via_push_relabel_new_labels", round);
    auto name_newName_ghost_vertices =
        ExchangeRepresentativesPush_Sort::execute(graph, parents, round);
    get_timer().stop("via_push_relabel_new_labels", round);
    get_timer().start("via_push_relabel_rename", round);
    EdgeRenamer::rename_edges(graph, parents, name_newName_ghost_vertices);

    get_timer().stop("via_push_relabel_rename", round);
  }
};

} // namespace hybridMST
