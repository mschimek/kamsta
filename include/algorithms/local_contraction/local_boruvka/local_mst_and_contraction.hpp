#pragma once

#include "parlay/hash_table.h"

#include "algorithms/base_case_mst_algos.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "local_filter_boruvka_with_halving.hpp"
#include "util/macros.hpp"

namespace hybridMST {

template <typename Map>
VId get_new_name_ghost_vertex(const VId& v, const Map& map) {
  auto it = growt::find(map, v);
  if (it == map.end()) {
    PRINT_WARNING_AND_ABORT("request vertex not contained in map");
  }
  return (*it).second;
}

inline VId get_new_name_local_vertex(VId v, const VertexRange_& vertex_range) {
  return v - vertex_range.v_begin;
}
//
template <typename Edges, typename Map>
void rename_edges(Edges& local_edges, Edges& non_local_edges,
                  const Map& new_names, const VertexRange_& range) {
  parallel_for(0, local_edges.size(), [&](std::size_t i) {
    auto& edge = local_edges[i];
    const auto src = get_new_name_local_vertex(edge.get_src(), range);
    const auto dst = get_new_name_local_vertex(edge.get_dst(), range);
    edge.set_src(src);
    edge.set_dst(dst);
  });
  parallel_for(0, non_local_edges.size(), [&](std::size_t i) {
    auto& edge = non_local_edges[i];
    const auto src = get_new_name_local_vertex(edge.get_src(), range);
    const auto dst = get_new_name_ghost_vertex(edge.get_dst(), new_names);
    edge.set_src(src);
    edge.set_dst(dst);
  });
}

/// @brief Map all vertices in (src, dst) in E to [0,n'] with n' = |{endpoints
/// in E}|
template <typename Edges> auto compactify_edges(const Edges& edges) {
  using EdgeType = typename Edges::value_type;
  if (edges.empty()) {
    return std::make_tuple(edges, std::uint64_t(0),
                           parlay::sequence<EdgeType>{},
                           parlay::sequence<EdgeType>{});
  }
  VertexRange_ vertex_range = create_vertex_range(edges);
  auto local_edges = parlay::filter(edges, [&](const auto& edge) {
    return is_true_local(edge, vertex_range) &&
           (edge.get_src() < edge.get_dst());
  });
  auto non_local_edges = parlay::filter(edges, [&](const auto& edge) {
    return !is_true_local(edge, vertex_range);
  });
  parlay::hashtable<parlay::hash_numeric<VId>> table(
      non_local_edges.size() * 1.1, parlay::hash_numeric<VId>{});
  parallel_for(0, non_local_edges.size(), [&](std::size_t i) {
    const auto& cur_edge = non_local_edges[i];
    table.insert(cur_edge.get_dst());
  });
  auto non_local_vertices = table.entries();
  const std::size_t map_size = non_local_vertices.size() * 1.1;
  growt::GlobalVIdMap<VId> map{map_size};
  parallel_for(0, non_local_vertices.size(), [&](std::size_t i) {
    const auto& ghost_vertex = non_local_vertices[i];
    growt::insert(
        map, ghost_vertex,
        vertex_range.n() +
            i); // we place the ghost vertices behind all local vertices which
                // are shifted to [0, num_local_vertices)
  });
  rename_edges(local_edges, non_local_edges, map, vertex_range);
  const std::uint64_t upper_bound_num_compact_vertices =
      vertex_range.n() + non_local_vertices.size();
  Edges renamed_edges(
      local_edges.size() +
      non_local_edges.size()); // as we filter out local edges (u,v) with u > v
                               // we cannot simply use edges.size()
  parallel_for(0, local_edges.size(),
               [&](std::size_t i) { renamed_edges[i] = local_edges[i]; });
  parallel_for(0, non_local_edges.size(), [&](std::size_t i) {
    renamed_edges[local_edges.size() + i] = non_local_edges[i];
  });
  return std::make_tuple(std::move(renamed_edges),
                         upper_bound_num_compact_vertices,
                         std::move(local_edges), std::move(non_local_edges));
}

template <typename T> class TD;
template <typename Edges>
Edges discard_local_non_mst_edges(const Edges& edges) {
  using EdgeType = typename Edges::value_type;

  get_timer().start("local_kernelization_boruvka_without_contract_compact", 0);
  auto [compactified_edges, compact_n, renamed_local_edges,
        renamed_non_local_edges] = compactify_edges(edges);
  get_timer().stop("local_kernelization_boruvka_without_contract_compact", 0);
  get_timer().start("local_kernelization_boruvka_without_contract_boruvka", 0);

  non_init_vector<GlobalEdgeId> mst_edge_ids(compact_n);
  non_init_vector<VId> parents(compact_n);
  parallel_for(0, compact_n, [&](std::size_t i) { parents[i] = i; });
  mpi::MPIContext ctx;
  constexpr bool debug = false;

  if constexpr (debug) {
    SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(edges););
    SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(compactified_edges););
  }
  const VId mst_edges_found =
      boruvka_filter_plain(VertexRange_(0, compact_n), Span(compactified_edges),
                           Span(parents), mst_edge_ids.data());
  // std::cout << "finished first contraction " << std::endl;
  mst_edge_ids.resize(mst_edges_found);
  get_timer().stop("local_kernelization_boruvka_without_contract_boruvka", 0);
  get_timer().start("local_kernelization_boruvka_without_contract_postprocess",
                    0);

  parlay::hashtable<parlay::hash_numeric<VId>> ids(mst_edge_ids.size() * 1.1,
                                                   parlay::hash_numeric<VId>{});
  parallel_for(0, mst_edge_ids.size(),
               [&](std::size_t i) { ids.insert(mst_edge_ids[i]); });
  renamed_local_edges =
      parlay::filter(renamed_local_edges, [&](const auto& edge) {
        // return edge.get_edge_id() == ids.find(edge.get_edge_id());
        return edge.get_edge_id() == ids.find(edge.get_edge_id());
      });

  const GlobalEdgeId first_edge_id =
      edges.empty() ? GLOBAL_EDGEID_UNDEFINED : edges.front().get_edge_id();
  Edges reduced_edges((renamed_local_edges.size() * 2ull) +
                      renamed_non_local_edges.size());
  parallel_for(0, renamed_local_edges.size(), [&](std::size_t i) {
    const auto& edge =
        edges[renamed_local_edges[i].get_edge_id() - first_edge_id];
    reduced_edges[i] = edge;
    auto flipped = edge;
    flipped.set_src(edge.get_dst());
    flipped.set_dst(edge.get_src());
    flipped.set_weight_and_edge_id(edge.get_weight(), edge.get_edge_id());
    reduced_edges[renamed_local_edges.size() + i] = flipped;
  });
  parallel_for(0, renamed_non_local_edges.size(), [&](std::size_t i) {
    const auto& edge =
        edges[renamed_non_local_edges[i].get_edge_id() - first_edge_id];
    reduced_edges[i + (2 * renamed_local_edges.size())] = edge;
  });

  get_timer().start("local_kernelization_boruvka_without_contract_sort", 0);
  // we need to sort here as the following steps expected a lexicographically
  // sorted edge list
  ips4o::parallel::sort(reduced_edges.begin(), reduced_edges.end(),
                        graphs::SrcDstOrder<EdgeType>{});
  get_timer().stop("local_kernelization_boruvka_without_contract_sort", 0);
  get_timer().stop("local_kernelization_boruvka_without_contract_postprocess",
                   0);
  return reduced_edges;
}

// debugging code
template <typename Edges, typename FilteredEdges, typename MstEdgeIds>
inline void
debug_discard_local_non_mst_edges(const Edges& edges,
                                  const FilteredEdges& renamed_local_edges,
                                  const FilteredEdges& renamed_non_local_edges,
                                  const MstEdgeIds& mst_edge_ids) {
  using EdgeType = typename Edges::value_type;
  const VertexRange_ org_vertex_range = create_vertex_range(edges);
  std::unordered_set<GlobalEdgeId> ids_std;
  for (const auto& id : mst_edge_ids) {
    ids_std.insert(id);
  }
  WEdgeList tmp;
  for (const auto& edge : edges) {
    tmp.emplace_back(edge.get_src(), edge.get_dst(), edge.get_weight());
  }
  Edges non_local_edges_expected;
  Edges local_edges_expected;
  for (const auto& edge : edges) {
    if (!is_true_local(edge, org_vertex_range)) {
      non_local_edges_expected.push_back(edge);
    } else {
      if (ids_std.find(edge.get_edge_id()) != ids_std.end()) {
        local_edges_expected.push_back(edge);
        auto flipped = edge;
        flipped.set_src(edge.get_dst());
        flipped.set_dst(edge.get_src());
        flipped.set_weight_and_edge_id(edge.get_weight(), edge.get_edge_id());
        local_edges_expected.push_back(flipped);
      }
    }
  }
  ips4o::parallel::sort(renamed_local_edges.begin(), renamed_local_edges.end(),
                        graphs::SrcDstWeightOrder<EdgeType>{});

  ips4o::parallel::sort(renamed_non_local_edges.begin(),
                        renamed_non_local_edges.end(),
                        graphs::SrcDstWeightOrder<EdgeType>{});
  mpi::MPIContext ctx;
  ctx.execute_in_order([&]() {
    if (local_edges_expected.size() != renamed_local_edges.size()) {
      std::cout << "unequal size" << std::endl;
    }
    for (std::size_t i = 0; i < local_edges_expected.size(); ++i) {
      if (local_edges_expected[i] != renamed_local_edges[i]) {
        std::cout << i << " exp: " << local_edges_expected[i]
                  << " act: " << renamed_local_edges[i] << std::endl;
      }
    }
    if (non_local_edges_expected.size() != renamed_non_local_edges.size()) {
      std::cout << "nonlocal unequal size" << std::endl;
    }
    for (std::size_t i = 0; i < non_local_edges_expected.size(); ++i) {
      if (non_local_edges_expected[i] != renamed_local_edges[i]) {
        std::cout << "non: " << i << " exp: " << non_local_edges_expected[i]
                  << " act: " << renamed_non_local_edges[i] << std::endl;
      }
    }
  });
  {
    auto expected_mst = local_kruskal(10000, tmp);
    Edges actual_mst;
    for (const auto& edge : edges) {
      if (ids_std.find(edge.get_edge_id()) != ids_std.end()) {
        actual_mst.push_back(edge);
      }
    }
    const std::uint64_t expected_sum = sum_edge_weights(expected_mst);
    const std::uint64_t actual_sum = sum_edge_weights(actual_mst);
    SEQ_EX(ctx, PRINT_VAR(expected_sum); PRINT_VAR(actual_sum););
  }
  Edges reduced_edges;
  for (const auto& edge : edges) {
    if (!is_true_local(edge, org_vertex_range)) {
      reduced_edges.push_back(edge);
    } else {
      if (ids_std.find(edge.get_edge_id()) != ids_std.end()) {
        reduced_edges.push_back(edge);
        auto flipped = edge;
        flipped.set_src(edge.get_dst());
        flipped.set_dst(edge.get_src());
        flipped.set_weight_and_edge_id(edge.get_weight(), edge.get_edge_id());
        reduced_edges.push_back(flipped);
      }
    }
  }
}
} // namespace hybridMST
