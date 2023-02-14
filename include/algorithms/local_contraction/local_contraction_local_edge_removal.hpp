#pragma once

#include "algorithms/gbbs_reimplementation.hpp"
#include "datastructures/distributed_graph_helper.hpp"
#include "definitions.hpp"
#include "util/utils.hpp"

namespace hybridMST {
namespace impl_local_mst_without_contraction {
template <typename Edge, typename LocalEdgeClassifier>
Span<Edge> partition_edges_rename_local_edges(Span<Edge> edges,
                                              VertexRange_ vertex_range,
                                              LocalEdgeClassifier&& local_edge_classifier) {
  auto begin_local_edges = std::partition(
      std::execution::par, edges.begin(), edges.end(),
      [&](const auto& edge) { return !local_edge_classifier(edge); });
  const std::size_t nb_cut_edges =
      static_cast<std::size_t>(std::distance(edges.begin(), begin_local_edges));
  const std::size_t nb_local_edges = edges.size() - nb_cut_edges;
  auto local_edges = Span(edges.data() + nb_cut_edges, nb_local_edges);
  tbb::parallel_for(TBB::IndexRange(0, local_edges.size()),
                    [&](TBB::IndexRange r) {
                      for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        auto& edge = local_edges[i];
                        src_ref(edge) -= vertex_range.v_begin;
                        dst_ref(edge) -= vertex_range.v_begin;
                      }
                    });
  return local_edges;
}
template <typename Edge>
Span<Edge> delete_local_non_mst_edges(std::size_t local_n,
                                      const Span<const WEdge> ref_edges,
                                      Span<Edge> edges,
                                      Span<Edge> local_edges) {
  const std::size_t nb_cut_edges = edges.size() - local_edges.size();
  std::vector<GlobalEdgeId> mst_edge_ids;
  const std::size_t nb_local_edges = local_edges.size();
  gbbs_reimplementation(local_n, local_edges, mst_edge_ids);
  auto replace_mst_edge_id_with_fwd_bwd_edge = [&](std::size_t i) {
    const GlobalEdgeId global_id = mst_edge_ids[i];
    const auto& local_edge_id = EdgeIdDistribution::get_local_id(global_id);
    const auto& mst_edge = ref_edges[local_edge_id];
    const auto forward_idx = nb_cut_edges + (2 * i);
    const auto backward_idx = nb_cut_edges + (2 * i) + 1;
    const VId src = src_ref(mst_edge);
    const VId dst = dst_ref(mst_edge);
    const Weight w = weight_ref(mst_edge);
    edges[forward_idx] = WEdgeId{src, dst, w, global_id};
    edges[backward_idx] = WEdgeId{dst, src, w, global_id};
  };
  tbb::parallel_for(TBB::IndexRange(0, mst_edge_ids.size()),
                    [&](TBB::IndexRange r) {
                      for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        replace_mst_edge_id_with_fwd_bwd_edge(i);
                      }
                    });
  const std::size_t nb_reduced_edges =
      nb_local_edges - (2 * mst_edge_ids.size());
  edges.resize(edges.size() - nb_reduced_edges);
  return edges;
}
} // namespace impl_local_mst_without_contraction
template <typename Edges, typename RefEdges>
void local_mst_without_contraction(VertexRange range, const RefEdges& ref_edges,
                                   Edges& edges, double threshold = 0.1) {
  mpi::MPIContext ctx;
  using EdgeType = typename Edges::value_type;
  const VertexRange_ vertex_range(range.first, range.second + 1);
  auto is_local_and_not_v_begin_v_end = [&](const EdgeType& edge) {
    return is_local<EdgeType>(edge, VertexRange_(range.first + 1, range.second));
  };
  const std::size_t nb_local_edges =
      std::count_if(std::execution::par, edges.begin(), edges.end(),
                    is_local_and_not_v_begin_v_end);
  const std::size_t nb_edges = edges.size();
  if ((nb_local_edges / static_cast<double>(nb_edges)) < threshold &&
      ctx.rank() == 0) {
    std::cout << "local edge removal: do nothing" << std::endl;
    return;
  }

  Span<EdgeType> local_edges =
      impl_local_mst_without_contraction::partition_edges_rename_local_edges(
          Span<EdgeType>(edges), vertex_range, is_local_and_not_v_begin_v_end);
  Span<EdgeType> all_edges =
      impl_local_mst_without_contraction::delete_local_non_mst_edges(
          vertex_range.n(),
          Span<const WEdge>(ref_edges.data(), ref_edges.size()),
          Span<EdgeType>(edges), local_edges);

  edges.resize(all_edges.size());
}
} // namespace hybridMST
