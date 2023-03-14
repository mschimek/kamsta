// This code is part of the project "Theoretically Efficient Parallel Graph
// Algorithms Can Be Fast and Scalable", presented at Symposium on Parallelism
// in Algorithms and Architectures, 2018.
// Copyright (c) 2018 Laxman Dhulipala, Guy Blelloch, and Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all  copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This file is based on https://github.com/ParAlg/gbbs/
// and was modified by Matthias Schimek

#pragma once

#include <algorithm>
#include <algorithms/base_case_mst_algos.hpp>
#include <numeric>
#include <random>

#include "gbbs_bridge/bridge.hpp"

#include "parlay/delayed_sequence.h"
#include "parlay/internal/binary_search.h"
#include "parlay/internal/get_time.h"
#include "parlay/io.h"
#include "parlay/monoid.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/range.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/utilities.h"

#include "definitions.hpp"
#include "mpi/context.hpp"
#include "util/atomic_ops.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

namespace hybridMST {

namespace gbbs {
constexpr const inline size_t kDefaultGranularity = 2048;
}  // namespace gbbs

template <typename T>
constexpr inline T MSD = ~(~T{0} >> 1);

namespace gbbs_boruvka {
///@brief Computes the minimum edge incident to the vertices stored in vertices.
///
/// The edges which are considered for this selection process are the ones whose
/// ids are stored in edge_ids.
template <typename Edges, typename MinEdges, typename EdgeIds,
          typename Vertices, typename VertexNormalization = Identity>
void compute_min_edges(
    std::size_t n, std::size_t m, MinEdges& min_edges, const EdgeIds& edge_ids,
    const Edges& edges, const Vertices& vertices,
    VertexNormalization&& normalizer = VertexNormalization{}) {
  using namespace parlay_bridge;
  const auto comp = [](const EdgeIdWeight& lhs, const EdgeIdWeight& rhs) {
    if (lhs.weight < rhs.weight) return true;
    if (lhs.weight == rhs.weight) return lhs.edge_id < rhs.edge_id;
    return false;
  };
  hybridMST::parallel_for(0, n, [&](size_t i) {
    VId v = vertices[i];
    min_edges[normalizer(v)] = EdgeIdWeight{LOCAL_EDGEID_UNDEFINED, WEIGHT_MAX};
  });
  hybridMST::parallel_for(0, m, [&](size_t i) {
    const LocalEdgeId e_id = edge_ids[i];
    const auto& edge = edges[e_id];
    const EdgeIdWeight id_weight{e_id, edge.get_weight()};
    hybridMST::write_min_gbbs(min_edges[normalizer(edge.get_src())], id_weight,
                              comp);
    hybridMST::write_min_gbbs(min_edges[normalizer(edge.get_dst())], id_weight,
                              comp);
  });
}

template <typename MinEdges, typename Edges, typename Vertices,
          typename Parents, typename RootsInfo, typename ExhaustionInfo,
          typename MstEdges>
void determine_mst_edges(std::size_t n, const MinEdges& min_edges,
                         const Edges& edges, const Vertices& vertices,
                         Parents& parents, RootsInfo& is_root,
                         ExhaustionInfo& exhausted, MstEdges& new_mst_edges) {
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const EdgeIdWeight& e = min_edges[v].load();
    if (e.edge_id == LOCAL_EDGEID_UNDEFINED) {
      // no more edges incident to v in this batch.
      exhausted[i] = true;
      is_root[i] = false;
      new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
    } else {
      const LocalEdgeId ind = e.edge_id;
      const auto& edge = edges[ind];
      const VId u = static_cast<VId>(edge.get_src()) ^
                    static_cast<VId>(edge.get_dst()) ^ v;
      // pick the lower endpoint as the root.
      if (u > v && ind == min_edges[u].load().edge_id) {
        parents[v] = v;
        is_root[i] = true;
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else {
        // v is satellite: hook onto u.
        parents[v] = u;
        is_root[i] = false;
        new_mst_edges[i] = ind;
      }
    }
  });
}

///@brief Adds the global edge ids of the mst edges compute during a boruvka
/// round to the mst-array.
///
/// Note that the ids contained in new_mst_edges is relative to the order within
/// the edges array.
template <typename NewMstEdgesIds, typename MstEdgesIds, typename Edges>
std::size_t add_new_mst_edges_to_mst_array(std::size_t n,
                                           std::size_t nb_mst_edges,
                                           NewMstEdgesIds& new_mst_edges,
                                           const Edges& edges,
                                           MstEdgesIds& mst) {
  const std::size_t prev_n_in_mst = nb_mst_edges;
  const std::size_t nb_added_mst_edges = parlay::filterf(
      new_mst_edges.begin(), mst + prev_n_in_mst, n,
      [](LocalEdgeId e_id) { return e_id != LOCAL_EDGEID_UNDEFINED; });

  // mpi::MPIContext ctx;
  nb_mst_edges += nb_added_mst_edges;
  hybridMST::parallel_for(0, nb_added_mst_edges, [&](size_t i) {
    // mst[prev_n_in_mst, nb_mst_edges) contains sizeof(LocalEdgeId) <=
    // sizeof(GlobalEdgeId) indices -> will be replaced with the global ids
    // if (ctx.rank() == 1) {
    //  std::cout << i << " " << edges[mst[prev_n_in_mst + i]] << std::endl;
    //}
    mst[prev_n_in_mst + i] = edges[mst[prev_n_in_mst + i]].get_edge_id();
  });
  return nb_mst_edges;
}

template <typename Vertices, typename Parents>
void update_parents(std::size_t n, const Vertices& vertices, Parents& parents) {
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    while (parents[v] != parents[parents[v]]) {
      parents[v] = parents[parents[v]];
    }
  });
}

template <typename Vertices, typename RootInfo>
std::size_t compactify_vertices(std::size_t n, Vertices& vertices,
                                Vertices& next_vertices,
                                const RootInfo& is_root) {
  auto vertices_input_slice = parlay_bridge::make_slice<VId>(vertices, n);
  auto vertices_output_slice = parlay_bridge::make_slice(next_vertices, n);
  n = parlay::pack_out(vertices_input_slice, is_root, vertices_output_slice);
  std::swap(vertices, next_vertices);
  return n;
}

template <typename EdgeIds, typename Edges, typename Parents,
          typename VertexNormalization = Identity>
void relabel_edges(std::size_t m, EdgeIds& edge_ids, Edges& edges,
                   const Parents& parents,
                   VertexNormalization&& normalizer = VertexNormalization{}) {
  parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity, [&](size_t i) {
    size_t e_id = edge_ids[i];
    auto& edge = edges[e_id];
    const VId u = static_cast<VId>(edge.get_src());
    const VId v = static_cast<VId>(edge.get_dst());
    const VId pu = parents[normalizer(u)];
    const VId pv = parents[normalizer(v)];
    if (u != pu || v != pv) {
      edge.set_src(pu);
      edge.set_dst(pv);
    }
    if (pu == pv) {
      edge_ids[i] |= MSD<LocalEdgeId>;  // mark self loops, so that they can be
                                        // filtered out
    }
  });
}

template <typename EdgeIds>
std::size_t filter_out_self_loops(std::size_t m, EdgeIds& edge_ids,
                                  EdgeIds& next_edge_ids) {
  auto is_not_self_loop = [&](size_t i) {
    return !(edge_ids[i] & MSD<LocalEdgeId>);
  };
  auto self_loop_input = parlay::delayed_seq<bool>(m, is_not_self_loop);
  auto edge_id_input_slice = parlay_bridge::make_slice(edge_ids, m);
  auto edge_id_output_slice = parlay_bridge::make_slice(next_edge_ids, m);

  m = parlay::pack_out(edge_id_input_slice, self_loop_input,
                       edge_id_output_slice);
  std::swap(edge_ids, next_edge_ids);
  return m;
}
template <typename T>
void debug_print(Span<T> data) {
  for (std::size_t i = 0; i < data.size(); ++i)
    std::cout << std::setw(10) << i << " " << data[i] << std::endl;
}
}  // namespace gbbs_boruvka

///@brief Computes the MST based on Boruvka's algorithm.
///
///@param edges Input edges based on which the MST is computed.
///@param vertices In/out param. Contains the vertex ids for which no mst edges
/// have been computed yet.
///@param tmp_vertices Only tmp storage needed during the algorithm.
///@param min_edges In param. In this storage the min edge ids (and weights) are
/// stored.
///@param parents In/out param. Contains the parents ids for each vertex.
///@param exhausted In/Out param. Will contain vertex ids which has no incident
/// edges.
///@param mst Out param. Will contain the global edge ids of the mst edges.
template <class EdgeType, class M, class P, class D>
inline std::size_t Boruvka(Span<EdgeType> edges, VId*& vertices,
                           VId*& tmp_vertices, M& min_edges, P& parents,
                           D& exhausted, size_t& n, GlobalEdgeId* mst) {
  constexpr bool is_output_activated = false;
  size_t m = edges.size();
  const size_t m_initial = m;
  // Note that the ids in edge_ids/next_edge_ids refer to the relative ordering
  // within the edges array not to the global ids of the edges in the edges
  // array.
  LocalEdgeId* edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
  parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity,
                              [&](size_t i) { edge_ids[i] = i; });
  LocalEdgeId* next_edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);

  // wrong type but needed to due to shortcoming in filterf
  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto is_root = parlay::sequence<bool>(n);

  // Stores edge indices that join the MinimumSpanningForest.
  size_t nb_mst_edges = 0;
  size_t round = 0;

  while (n > 1 && m > 0) {
    if constexpr (is_output_activated) {
      gbbs_boruvka::debug_print(Span(edge_ids, m));
      gbbs_boruvka::debug_print(edges);
    }
    gbbs_boruvka ::compute_min_edges(n, m, min_edges, edge_ids, edges,
                                     vertices);
    gbbs_boruvka::determine_mst_edges(n, min_edges, edges, vertices, parents,
                                      is_root, exhausted, new_mst_edges);
    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, edges, mst);
    gbbs_boruvka::update_parents(n, vertices, parents);
    n = gbbs_boruvka::compactify_vertices(n, vertices, tmp_vertices, is_root);
    gbbs_boruvka::relabel_edges(m, edge_ids, edges, parents);
    m = gbbs_boruvka::filter_out_self_loops(m, edge_ids, next_edge_ids);
    round++;
  }

  parlay_bridge::free_array(edge_ids, m_initial);
  parlay_bridge::free_array(next_edge_ids, m_initial);
  return nb_mst_edges;
}

template <typename Container>
Span<typename Container::value_type> partition(std::size_t n,
                                               Container& container) {
  if (container.empty()) return Span(container.data(), 0);
  if (3 * n / 2 > container.size())
    return Span(container.data(), container.size());
  const std::size_t nb_samples =
      std::max(static_cast<std::size_t>(100ull),
               static_cast<std::size_t>(0.01 * container.size()));
  if (container.size() < nb_samples)
    return Span(container.data(), container.size());
  std::mt19937 mt(0);
  std::uniform_int_distribution<std::size_t> dist(0, container.size() - 1);
  std::vector<Weight> samples(nb_samples);
  for (std::size_t i = 0; i < nb_samples; ++i) {
    samples[i] = container[dist(mt)].get_weight();
  }
  std::sort(samples.begin(), samples.end());
  const double ratio =
      std::min(static_cast<double>(3 * n) / (2 * container.size()), 1.0);
  const std::size_t sample_idx_raw = nb_samples * ratio;
  const std::size_t sample_idx = (sample_idx_raw >= samples.size())
                                     ? (samples.size() - 1)
                                     : sample_idx_raw;
  const Weight pivot = samples[sample_idx];
  const auto light_edges = parlay::filter(
      container, [&](const auto& edge) { return edge.get_weight() <= pivot; });
  const auto heavy_edges = parlay::filter(
      container, [&](const auto& edge) { return edge.get_weight() > pivot; });

  parallel_for(0, light_edges.size(),
               [&](std::size_t i) { container[i] = light_edges[i]; });
  parallel_for(0, heavy_edges.size(), [&](std::size_t i) {
    container[i + light_edges.size()] = heavy_edges[i];
  });
  const auto size_first_partition = light_edges.size();
  return Span(container.data(), size_first_partition);
}

template <typename OutContainer, typename Container,
          typename T = typename Container::value_type>
OutContainer get_light_edges(std::size_t n, const T& pivot,
                             Container& container) {
  const auto is_light = [&](const T& edge) { edge.get_weight() <= pivot; };
  const std::size_t num_light_edges = parlay::count_if(container, is_light);
  OutContainer light_edges(num_light_edges);
  filter(container, light_edges, is_light);
  return light_edges;
}

template <typename Edges, typename Parents>
void debug_print(Edges&& edges, Parents&& parents, const std::string desc) {
  std::cout << desc << std::endl;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    auto& e = edges[i];
    std::cout << i << " u:" << e.get_src() << " v:" << e.get_dst()
              << " weight: " << e.get_weight()
              << "  p[u]:" << parents[e.get_src()]
              << " p[v]:" << parents[e.get_dst()] << " id: " << e.get_edge_id()
              << std::endl;
  }
}

inline std::vector<GlobalEdgeId> MinimumSpanningForest(std::size_t n,
                                                       Span<WEdgeId>& edges_) {
  // get_timer().start("init_edges");
  auto edges_in = parlay_bridge::make_slice(edges_.data(), edges_.size());
  auto edges = filter(edges_in, [](const auto& edge) {
    return edge.get_src() > edge.get_dst();
  });
  // get_timer().stop("init_edges");
  // get_timer().start("init");
  constexpr bool is_output_activated = false;
  non_init_vector<std::atomic<EdgeIdWeight>> min_edges(n);
  auto parents =
      parlay::sequence<VId>::from_function(n, [](size_t i) { return i; });
  auto exhausted =
      parlay::sequence<bool>::from_function(n, [](size_t) { return false; });
  std::vector<GlobalEdgeId> mst_edge_global_ids(n);

  const std::size_t initial_n = n;
  std::size_t n_active = n;
  VId* vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  VId* next_vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  parlay_bridge::parallel_for(0, n, gbbs::kDefaultGranularity,
                              [&](size_t i) { vtxs[i] = i; });

  std::size_t round = 0;
  std::size_t nb_mst_edges_computed = 0;

  // get_timer().stop("init");
  // get_timer().start("rounds");
  while (edges.size() > 0) {
    std::size_t m = edges.size();
    if constexpr (is_output_activated) {
      std::cout << "\n";
      std::cout << "round = " << round << " n_active = " << n_active
                << " m = " << m
                << " MinimumSpanningForest size = " << nb_mst_edges_computed
                << "\n";
    }

    // relabel edges
    if (round > 0) {
      parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity,
                                  [&](size_t i) {
                                    auto& e = edges[i];
                                    const VId u = e.get_src();
                                    const VId v = e.get_dst();
                                    e.set_src(parents[u]);
                                    e.set_dst(parents[v]);
                                  });
    }
    // find a prefix of lowest-weight edges.
    // get_timer().start("partition", round);
    auto E = round < 4 ? partition(n_active, edges)
                       : Span(edges.data(), edges.size());
    // get_timer().stop("partition", round);

    if constexpr (is_output_activated) {
      debug_print(edges, parents, "Edges");
      debug_print(E, parents, "E");
    }

    size_t n_in_mst =
        Boruvka(E, vtxs, next_vtxs, min_edges, parents, exhausted, n_active,
                mst_edge_global_ids.data() + nb_mst_edges_computed);
    if constexpr (is_output_activated) {
      std::cout << "Found MST edge ids (nb: " << n_in_mst << ")"
                << "\n";
      std::cout << "n_active " << n_active << "\n";
    }
    nb_mst_edges_computed += n_in_mst;

    auto vtx_range = parlay_bridge::make_slice(vtxs + n_active, vtxs + n);
    n_active +=
        parlay::pack_index_out(parlay_bridge::make_slice(exhausted), vtx_range);

    parlay_bridge::parallel_for(0, n, gbbs::kDefaultGranularity, [&](size_t i) {
      if (exhausted[i]) exhausted[i] = false;
    });

    // pointer jump: vertices that were made inactive could have had their
    // parents change.
    parlay_bridge::parallel_for(0, n, gbbs::kDefaultGranularity, [&](size_t i) {
      while (parents[i] != parents[parents[i]]) {
        parents[i] = parents[parents[i]];
      }
    });

    // get_timer().start("remove_edge", round);
    //  pack out all edges in the graph that are shortcut by the added edges
    auto is_edge_not_self_loop = [&](const auto& edge) {
      auto c_src = parents[edge.get_src()];
      auto c_dst = parents[edge.get_dst()];
      return c_src != c_dst;
    };
    edges =
        parlay::filter(parlay_bridge::make_slice(edges.data(), edges.size()),
                       is_edge_not_self_loop);

    // get_timer().stop("remove_edge", round);

    if constexpr (is_output_activated) {
      std::cout << "After filter, m is now " << edges.size() << "\n";
      std::cout << "n: " << n << " n_active: " << n_active << " m is now "
                << edges.size() << "\n";
    }
    round++;
  }
  if constexpr (is_output_activated) {
    std::cout << "#edges in output mst: " << nb_mst_edges_computed << "\n";
  }

  mst_edge_global_ids.resize(nb_mst_edges_computed);
  if constexpr (is_output_activated) {
    std::cout << "mst edges" << std::endl;
    for (std::size_t i = 0; i < mst_edge_global_ids.size(); ++i) {
      const auto& elem = mst_edge_global_ids[i];
      std::cout << elem << std::endl;
    }
  }

  parlay_bridge::free_array(vtxs, initial_n);
  parlay_bridge::free_array(next_vtxs, initial_n);
  return mst_edge_global_ids;
}

inline void gbbs_reimplementation(std::size_t n, Span<WEdgeId> edges,
                                  std::vector<GlobalEdgeId>& mst_edge_ids) {
  mpi::MPIContext ctx;
  mst_edge_ids = MinimumSpanningForest(n, edges);
  // std::cout << "found mst size: " << mst_edge_ids.size() << std::endl;
}
}  // namespace hybridMST
