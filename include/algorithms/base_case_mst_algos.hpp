#pragma once

#include <algorithm>
#include <numeric>

#include "ips4o/ips4o.hpp"

#include "datastructures/union_find.hpp"
#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/context.hpp"
#include "mpi/gather.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

// #include "interface.hpp"

namespace hybridMST {

template <typename Compare = WeightOrder<WEdgeId>>
inline void local_kruskal(std::vector<WEdgeId>& edges,
                          std::vector<GlobalEdgeId>& global_edge_ids) {
  std::sort(edges.begin(), edges.end(), Compare{});
  MapBasedUnionFind uf;
  for (auto& elem : edges) {
    if (uf.find(elem.get_src()) == uf.find(elem.get_dst())) continue;
    global_edge_ids.push_back(elem.get_global_id());
    uf.unify(elem.get_src(), elem.get_dst());
  }
}

// template <typename EdgeType, typename Compare = WeightOrder<EdgeType>,
//           bool execute_in_parallel = true>
// void gbbs_mst(std::size_t n, std::vector<EdgeType>& edges,
//               std::vector<GlobalEdgeId>& global_edge_ids) {
//
//   hybridMST::get_timer().start("gen_gbbs_graph");
//   auto graph = gbbs::generate_graph(n, edges);
//   hybridMST::get_timer().stop("gen_gbbs_graph");
//
//   hybridMST::get_timer().start("gbbs");
//   auto mst_ids = compute_minimum_spanning_forest(graph);
//   hybridMST::get_timer().stop("gbbs");
//
//   hybridMST::get_timer().start("combine");
//   for (const auto& mst_id : mst_ids) {
//     global_edge_ids.push_back(mst_id);
//   }
//   hybridMST::get_timer().stop("combine");
// }

template <typename EdgeType, typename Compare = WeightOrder<EdgeType>,
          bool execute_in_parallel = true>
inline void local_kruskal(std::size_t n, std::vector<EdgeType>& edges,
                          std::vector<GlobalEdgeId>& global_edge_ids) {
  auto remove_f = [](const EdgeType& edge) {
    return edge.get_src() > edge.get_dst();
  };
  if constexpr (execute_in_parallel) {
    auto end = std::remove_if(edges.begin(), edges.end(), remove_f);
    edges.erase(end, edges.end());
    ips4o::parallel::sort(edges.begin(), edges.end(), Compare{});
  } else {
    auto end = std::remove_if(edges.begin(), edges.end(), remove_f);
    edges.erase(end, edges.end());
    ips4o::sort(edges.begin(), edges.end(), Compare{});
  }
  UnionFind uf(n);
  for (auto& elem : edges) {
    if (uf.find(elem.get_src()) == uf.find(elem.get_dst())) continue;
    global_edge_ids.push_back(elem.global_id);
    uf.unify(elem.get_src(), elem.get_dst());
  }
}

template <typename EdgeType, typename Compare = WeightOrder<EdgeType>>
inline void local_kruskal(std::size_t n, std::vector<EdgeType>& edges,
                          std::vector<GlobalEdgeId>& global_edge_ids,
                          execution::parallel) {
  local_kruskal<EdgeType, Compare, true>(n, edges, global_edge_ids);
}

template <typename EdgeType, typename Compare = WeightOrder<EdgeType>>
inline void local_kruskal(std::size_t n, std::vector<EdgeType>& edges,
                          std::vector<GlobalEdgeId>& global_edge_ids,
                          execution::sequential) {
  local_kruskal<EdgeType, Compare, false>(n, edges, global_edge_ids);
}

template <typename Edges,
          typename Compare = WeightSrcDstOrder<typename Edges::value_type>>
inline Edges local_kruskal(Edges& edges) {
  mpi::MPIContext ctx;
  if (ctx.rank() != 0) {
    return {};
  }
  using EdgeType = typename Edges::value_type;
  ips4o::parallel::sort(edges.begin(), edges.end(), Compare{});
  auto it = std::max_element(edges.begin(), edges.end(),
                             SrcDstWeightOrder<EdgeType>{});
  std::size_t n = it->get_src() + 1;
  Edges mst_edges;
  UnionFind uf(n);

  for (auto& elem : edges) {
    if (uf.find(elem.get_src()) == uf.find(elem.get_dst())) continue;
    mst_edges.push_back(elem);
    uf.unify(elem.get_src(), elem.get_dst());
  }
  return mst_edges;
}

template <typename Compare = WeightOrder<WEdge>,
          bool execute_in_parallel = true>
inline WEdgeList local_kruskal(std::size_t n, WEdgeList& edges) {
  if constexpr (execute_in_parallel) {
    ips4o::parallel::sort(edges.begin(), edges.end(), Compare{});
  } else {
    ips4o::sort(edges.begin(), edges.end(), Compare{});
  }
  WEdgeList mst_edges;
  std::cout << n << std::endl;
  UnionFind uf(n);
  for (auto& elem : edges) {
    if (uf.find(elem.get_src()) == uf.find(elem.get_dst())) continue;
    mst_edges.push_back(elem);
    uf.unify(elem.get_src(), elem.get_dst());
  }
  return mst_edges;
}

template <typename Edges>
inline void gather_mst(Edges& edges,
                       std::vector<GlobalEdgeId>& global_edge_ids) {
  mpi::MPIContext ctx;
  const PEID root = 0;
  auto recv_edges =
      hybridMST::mpi::gatherv(edges.data(), edges.size(), root, ctx);
  if (ctx.rank() == root) local_kruskal(recv_edges, global_edge_ids);
}

template <typename Edges>
inline Edges gather_mst(const Edges& edges) {
  mpi::MPIContext ctx;
  const PEID root = 0;
  auto recv_edges =
      hybridMST::mpi::gatherv(edges.data(), edges.size(), root, ctx);
  return local_kruskal(recv_edges);
}

}  // namespace hybridMST
