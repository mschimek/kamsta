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

#include "gbbs_bridge/bridge.hpp"

#include <algorithm>
#include <numeric>
#include <random>

#include "algorithms/base_case_mst_algos.hpp"
#include "datastructures/distributed_array.hpp"
#include "datastructures/distributed_parent_array.hpp"
#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/context.hpp"
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
#include "util/atomic_ops.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

namespace hybridMST {

namespace gbbs_boruvka_distributed {

// parallel loop from start (inclusive) to end (exclusive) running
// function f.
//    f should map long to void.
//    granularity is the number of iterations to run sequentially
//      if 0 (default) then the scheduler will decide
//    conservative uses a safer scheduler
template <typename F>
inline void parallel_for(size_t start, size_t end, long granularity, F f,
                         bool conservative = false) {
  return parlay_bridge::parallel_for(start, end, f, granularity, conservative);
}

template <typename T> constexpr inline T MSD = ~(~T{0} >> 1);
constexpr const inline size_t kDefaultGranularity = 2048;

template <typename EdgeType> struct WeightSrcDstOrder {
  EdgeType operator()(EdgeType& lhs, WEdgeId& rhs) {

    const auto& lhs_weight = lhs.get_weight();
    const auto& rhs_weight = rhs.get_weight();
    if (lhs_weight == rhs_weight) {
      const auto& lhs_src = lhs.get_src();
      const auto& lhs_dst = lhs.get_dst();
      const auto& rhs_src = rhs.get_src();
      const auto& rhs_dst = rhs.get_dst();
      const bool is_lhs_smaller =
          std::tie(lhs_src, lhs_dst) < std::tie(rhs_src, rhs_dst);
      return is_lhs_smaller ? lhs : rhs;
    }
    const bool is_lhs_smaller = lhs_weight < rhs_weight;
    return is_lhs_smaller ? lhs : rhs;
  }
};

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
  mpi::MPIContext ctx;

  parlay_bridge::parallel_for(0, n, [&](size_t i) {
    VId v = vertices[i];
    min_edges[normalizer(v)] =
        EdgeIdWeightDst{LOCAL_EDGEID_UNDEFINED, WEIGHT_INF, VID_UNDEFINED};
  });
  auto comp = [](const EdgeIdWeightDst& lhs, const EdgeIdWeightDst& rhs) {
    if (lhs.weight == rhs.weight) {
      return lhs.dst < rhs.dst;
    }
    return lhs.weight < rhs.weight;
  };
  parlay_bridge::parallel_for(0, m, kDefaultGranularity, [&](size_t i) {
    const LocalEdgeId e_id = edge_ids[i];
    const auto& edge = edges[e_id];
    const VId& dst = edge.get_dst();
    const EdgeIdWeightDst id_weight_dst{e_id, edge.get_weight(), dst};
    hybridMST::write_min(min_edges[edge.get_src()], id_weight_dst, comp);
  });
  // SEQ_EX(ctx, PRINT_VECTOR(min_edges););
}

template <typename Edges, typename MinEdges, typename Vertices,
          typename AllreduceInOut>
void allreduce_min(std::size_t n, const Edges& edges, const MinEdges& min_edges,
                   const Vertices& vertices, AllreduceInOut& allreduce_inout) {
  mpi::MPIContext ctx;

  WEdgeId sentinel = WEdgeId{VID_UNDEFINED, VID_UNDEFINED, WEIGHT_INF,
                             GLOBAL_EDGEID_UNDEFINED};
  allreduce_inout.resize(n);
  parlay_bridge::parallel_for(0, n, kDefaultGranularity,
                              [&](size_t i) { allreduce_inout[i] = sentinel; });
  parlay_bridge::parallel_for(0, n, [&](size_t i) {
    const VId src_ = vertices[i];
    const auto& elem = min_edges[src_].load();
    if (is_defined(elem.edge_id)) {
      const auto& edge = edges[elem.edge_id];
      WEdgeId reduce_edge;
      reduce_edge.set_src(src_);
      reduce_edge.set_dst(edge.get_dst());
      reduce_edge.set_weight_and_edge_id(edge.get_weight(), edge.get_edge_id());
      allreduce_inout[i] = reduce_edge;
    }
  });

  allreduce_inout = mpi::allreduce(
      allreduce_inout, WeightSrcDstOrder<WEdgeId>{}, mpi::MPIContext{});
  //  SEQ_EX(
  //      ctx, std::cout << "allreduce_inout" << std::endl;
  //      for (std::size_t i = 0; i < n;
  //           ++i) { std::cout << allreduce_inout[i] << std::endl; };);
}

template <typename AllreduceInOut, typename Vertices,
          typename Parents, typename RootsInfo, typename ExhaustionInfo,
          typename MstEdges>
void determine_mst_edges(std::size_t n, const AllreduceInOut& min_edges,
                         const Vertices& vertices,
                         Parents& parents, RootsInfo& is_root,
                         ExhaustionInfo& exhausted, MstEdges& new_mst_edges) {
  mpi::MPIContext ctx;
  const VId v_begin = static_cast<VId>(ctx.rank() * n) / ctx.size();
  VId v_end = static_cast<VId>((ctx.rank() + 1) * n) / ctx.size();
  if (ctx.rank() + 1 == ctx.size()) {
    v_end = n;
  }
  static_assert(std::is_same_v<WEdgeId, typename AllreduceInOut::value_type>,
                "MPI_Allreduce is done in WEdgeId type");
  parlay_bridge::parallel_for(0, n, kDefaultGranularity, [&](size_t i) {
    const VId src = vertices[i];
    const WEdgeId& e = min_edges[src];
    if (!is_defined(e.get_src())) {
      // no more edges incident to v in this batch.
      exhausted[src] = true;
      is_root[i] = false;
      new_mst_edges[i] = GLOBAL_EDGEID_UNDEFINED;
    } else {
      // if (src != src_ref(e)) {
      //   std::cout << "abort" << i << " " << src << " " << src_ref(e) << " "
      //   << e
      //             << std::endl;
      //   std::abort();
      // }
      const VId dst = e.get_dst();
      const VId dst_dst = min_edges[dst].get_dst();
      // pick the lower endpoint as the root.
      if (dst > src && src == dst_dst) {
        parents[src] = src;
        is_root[i] = true;
        new_mst_edges[i] = GLOBAL_EDGEID_UNDEFINED;
      } else {
        // v is satellite: hook onto u.
        parents[src] = dst;
        is_root[i] = false;
        if (v_begin <= i && i < v_end) {
          new_mst_edges[i] = e.get_edge_id();
          // if(e.weight == 9517) {
          //   std::cout << e << std::endl;
          // }
        } else {
          new_mst_edges[i] = GLOBAL_EDGEID_UNDEFINED;
        }
      }
    }
  });
}

///@brief Adds the global edge ids of the mst edges compute during a boruvka
/// round to the mst-array.
///
/// Note that the ids contained in new_mst_edges is relative to the order within
/// the edges array.
template <typename NewMstEdgesIds, typename MstEdgesIds>
std::size_t add_new_mst_edges_to_mst_array(std::size_t n,
                                           std::size_t nb_mst_edges,
                                           NewMstEdgesIds& new_mst_edges,
                                           MstEdgesIds& mst) {
  const std::size_t prev_n_in_mst = nb_mst_edges;
  const std::size_t nb_added_mst_edges = parlay::filterf(
      new_mst_edges.begin(), mst + prev_n_in_mst, n,
      [](GlobalEdgeId e_id) { return e_id != GLOBAL_EDGEID_UNDEFINED; });

  nb_mst_edges += nb_added_mst_edges;
  parlay_bridge::parallel_for(0, nb_added_mst_edges, [&](size_t i) {
    mst[prev_n_in_mst + i] = mst[prev_n_in_mst + i];
  });
  return nb_mst_edges;
}

template <typename Vertices, typename Parents>
void update_parents(std::size_t n, const Vertices& vertices, Parents& parents) {
  parlay_bridge::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    while (parents[v] != parents[parents[v]]) {
      parents[v] = parents[parents[v]].load();
    }
  });
}

template <typename Vertices, typename RootInfo>
std::size_t compactify_vertices(std::size_t n, Vertices& vertices,
                                Vertices& next_vertices,
                                const RootInfo& is_root) {
  auto vertices_input_slice = parlay_bridge::make_slice<VId>(vertices, n);
  auto vertices_output_slice = parlay_bridge::make_slice(next_vertices, n);
  n = ::parlay::pack_out(vertices_input_slice, is_root, vertices_output_slice);
  std::swap(vertices, next_vertices);
  return n;
}

template <typename EdgeIds, typename Edges, typename Parents,
          typename VertexNormalization = Identity>
void relabel_edges(std::size_t m, const EdgeIds& edge_ids, Edges& edges,
                   const Parents& parents,
                   VertexNormalization&& normalizer = VertexNormalization{}) {
  parlay_bridge::parallel_for(0, m, kDefaultGranularity, [&](size_t i) {
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
      edge_ids[i] |= MSD<LocalEdgeId>; // mark self loops, so that they can be
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
template <typename T> void debug_print(Span<T> data, const std::string desc) {
  std::cout << desc << std::endl;
  for (std::size_t i = 0; i < data.size(); ++i)
    std::cout << std::setw(10) << i << " " << data[i] << std::endl;
}
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
  mpi::MPIContext ctx;
  constexpr bool is_output_activated = false;
  size_t m = edges.size();
  const size_t m_initial = m;
  // Note that the ids in edge_ids/next_edge_ids refer to the relative ordering
  // within the edges array not to the global ids of the edges in the edges
  // array.
  LocalEdgeId* edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
  parlay_bridge::parallel_for(0, m,
                              gbbs_boruvka_distributed::kDefaultGranularity,
                              [&](size_t i) { edge_ids[i] = i; });
  LocalEdgeId* next_edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
  std::vector<WEdgeId> allreduce_inout(n);
  std::vector<WEdgeId> allreduce_inout_inital_indices(n);

  // wrong type but needed to due to shortcoming in filterf
  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto is_root = parlay::sequence<bool>(n);

  // Stores edge indices that join the MinimumSpanningForest.
  size_t nb_mst_edges = 0;
  size_t round = 0;

  while (n > 1) {
    if constexpr (is_output_activated) {
      ctx.execute_in_order([&]() {
        gbbs_boruvka_distributed::debug_print(Span(edge_ids, m), "edge_ids");
        gbbs_boruvka_distributed::debug_print(edges, "edges");
        std::cout << "n: " << n << " m: " << m << std::endl;
      });
    }
    get_timer().start("min_edges", round);
    gbbs_boruvka_distributed::compute_min_edges(n, m, min_edges, edge_ids,
                                                edges, vertices);
    get_timer().stop("min_edges", round);
    get_timer().start("allreduce_edges", round);
    gbbs_boruvka_distributed::allreduce_min(n, edges, min_edges, vertices,
                                            allreduce_inout);

    parlay_bridge::parallel_for(
        0, n, gbbs_boruvka_distributed::kDefaultGranularity, [&](size_t i) {
          const VId v = vertices[i];
          allreduce_inout_inital_indices[v] = allreduce_inout[i];
        });
    get_timer().stop("allreduce_edges", round);

    get_timer().start("determine_mst_edges", round);
    gbbs_boruvka_distributed::determine_mst_edges(
        n, allreduce_inout_inital_indices, vertices, parents, is_root,
        exhausted, new_mst_edges);
    get_timer().stop("determine_mst_edges", round);
    get_timer().start("add_new_mst", round);
    nb_mst_edges = gbbs_boruvka_distributed::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, mst);
    get_timer().stop("add_new_mst", round);
    get_timer().start("update_parents", round);
    gbbs_boruvka_distributed::update_parents(n, vertices, parents);
    get_timer().stop("update_parents", round);
    get_timer().start("compactify", round);
    n = gbbs_boruvka_distributed::compactify_vertices(n, vertices, tmp_vertices,
                                                      is_root);
    get_timer().stop("compactify", round);
    get_timer().start("relabel", round);
    gbbs_boruvka_distributed::relabel_edges(m, edge_ids, edges, parents);
    get_timer().stop("relabel", round);
    get_timer().start("filter_out", round);
    m = gbbs_boruvka_distributed::filter_out_self_loops(m, edge_ids,
                                                        next_edge_ids);
    get_timer().stop("filter_out", round);
    round++;
  }

  parlay_bridge::free_array(edge_ids, m_initial);
  parlay_bridge::free_array(next_edge_ids, m_initial);
  return nb_mst_edges;
}
} // namespace gbbs_boruvka_distributed

template <class Edges>
void dense_boruvka_allreduce(const std::size_t n, Edges& edges_,
                             std::vector<GlobalEdgeId>& mst_edge_global_ids) {
  mpi::MPIContext ctx;
  non_init_vector<std::atomic<EdgeIdWeightDst>> min_edges(n);
  auto parents =
      parlay::sequence<std::atomic<VId>>::from_function(n, [](size_t i) { return i; });
  auto exhausted =
      parlay::sequence<bool>::from_function(n, [](size_t) { return false; });
  mst_edge_global_ids.resize(n);

  std::size_t n_active = n;
  VId* vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  VId* next_vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  parlay_bridge::parallel_for(0, n,
                              gbbs_boruvka_distributed::kDefaultGranularity,
                              [&](size_t i) { vtxs[i] = i; });

  std::size_t nb_mst_edges_computed = 0;
  auto E = Span(edges_.data(), edges_.size());
  get_timer().start("boruvka_", 0);
  size_t n_in_mst = gbbs_boruvka_distributed::Boruvka(
      E, vtxs, next_vtxs, min_edges, parents, exhausted, n_active,
      mst_edge_global_ids.data() + nb_mst_edges_computed);
  get_timer().stop("boruvka_", 0);

  parlay_bridge::free_array(vtxs, n);
  parlay_bridge::free_array(next_vtxs, n);
  nb_mst_edges_computed += n_in_mst;
  mst_edge_global_ids.resize(nb_mst_edges_computed);
  // SEQ_EX(ctx, PRINT_VECTOR(mst_edge_global_ids););
}

template <class Edges>
void dense_boruvka_allreduce(const std::size_t n, Edges& edges_,
                             std::vector<GlobalEdgeId>& mst_edge_global_ids,
                             const non_init_vector<VId>& vertex_to_org_vertex,
                             ParentArray& parent_array) {
  mpi::MPIContext ctx;
  non_init_vector<std::atomic<EdgeIdWeightDst>> min_edges(n);
  auto parents =
      parlay::sequence<std::atomic<VId>>::from_function(n, [](size_t i) { return i; });
  auto exhausted =
      parlay::sequence<bool>::from_function(n, [](size_t) { return false; });
  mst_edge_global_ids.resize(n);

  const std::size_t initial_n = n;
  std::size_t n_active = n;
  VId* vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  VId* next_vtxs = parlay_bridge::new_array_no_init<VId>(n_active);
  parlay_bridge::parallel_for(0, n,
                              gbbs_boruvka_distributed::kDefaultGranularity,
                              [&](size_t i) { vtxs[i] = i; });

  std::size_t nb_mst_edges_computed = 0;
  auto E = Span(edges_.data(), edges_.size());
  get_timer().start("boruvka_", 0);
  size_t n_in_mst = gbbs_boruvka_distributed::Boruvka(
      E, vtxs, next_vtxs, min_edges, parents, exhausted, n_active,
      mst_edge_global_ids.data() + nb_mst_edges_computed);
  get_timer().stop("boruvka_", 0);

  if (ctx.rank() == 0) {
    non_init_vector<ParentArray::VertexParent> pa_updates(initial_n);
    for (std::size_t i = 0; i < initial_n; ++i) {
      VId v_org = vertex_to_org_vertex[i];
      VId parent = parents[i];
      VId parent_org = vertex_to_org_vertex[parent];
      pa_updates[i] = ParentArray::VertexParent{v_org, parent_org};
      // std::cout << i << " v_org " << v_org << " parent: " << parent
      //           << " parent_org: " << parent_org << std::endl;
    }
    // PRINT_VECTOR(pa_updates);
    parent_array.update(pa_updates);
  } else {
    std::vector<ParentArray::VertexParent> pa_updates(0);
    parent_array.update(pa_updates);
  }

  parlay_bridge::free_array(vtxs, n);
  parlay_bridge::free_array(next_vtxs, n);
  nb_mst_edges_computed += n_in_mst;
  mst_edge_global_ids.resize(nb_mst_edges_computed);
  // SEQ_EX(ctx, PRINT_VECTOR(mst_edge_global_ids););
}

} // namespace hybridMST
