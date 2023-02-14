#pragma once

#include <tbb/parallel_for.h>

#include "datastructures/distributed_graph_helper.hpp"
#include "definitions.hpp"
#include "mpi/context.hpp"
#include "util/utils.hpp"

namespace hybridMST {
struct LocalKernelization {
  template <typename Edge, typename WEdgeIdType>
  static non_init_vector<WEdgeIdType> execute(VertexRange range,
                                          const Span<Edge>& edges) {
    mpi::MPIContext ctx;
    const auto edge_id_offset = mpi::exscan_sum(edges.size(), ctx, 0ul);
    non_init_vector<WEdgeIdType> augmented_edges(edges.size());
    tbb::parallel_for(TBB::IndexRange(0, edges.size()), [&](TBB::IndexRange r) {
      for (LocalEdgeId i = r.begin(); i != r.end(); ++i) {
        const auto& edge = edges[i];
        WEdgeIdType augmented_edge;
        augmented_edge.set_src(edge.get_src());
        augmented_edge.set_dst(edge.get_dst());
        augmented_edge.set_weight(edge.get_weight());
        augmented_edge.set_edge_id(edge_id_offset + i);
        augmented_edges[i] = augmented_edge;
      }
    });
    return augmented_edges;
  }
};

//struct LocalKernelization_LocalEdgeRemoval {
//  template <typename Edge, typename WEdgeIdType>
//  static non_init_vector<WEdgeIdType> execute(VertexRange range,
//                                          const Span<Edge>& edges) {
//    mpi::MPIContext ctx;
//    non_init_vector<WEdgeIdType> augmented_edges(edges.size());
//    tbb::parallel_for(TBB::IndexRange(0, edges.size()), [&](TBB::IndexRange r) {
//      for (LocalEdgeId i = r.begin(); i != r.end(); ++i) {
//        const auto& edge = edges[i];
//        augmented_edges[i] =
//            WEdgeIdType{edge, EdgeIdDistribution::get_id(i, ctx.rank())};
//      }
//    });
//    // SEQ_EX(ctx, PRINT_VECTOR(augmented_edges););
//    local_mst_without_contraction(range, edges, augmented_edges);
//    return augmented_edges;
//  }
//};
} // namespace hybridMST
