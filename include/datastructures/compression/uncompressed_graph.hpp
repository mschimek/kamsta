#pragma once

#include "definitions.hpp"
#include "shared_mem_parallel.hpp"
#include "util/macros.hpp"

namespace hybridMST {
template <typename WEdgeType_, typename WEdgeIdType_> class UncompressedGraph {
public:
  using WEdgeType = WEdgeType_;
  using WEdgeIdType = WEdgeIdType_;
  using WEdgeContainer = non_init_vector<WEdgeType>;
  using WEdgeIdContainer = non_init_vector<WEdgeIdType>;

  template <typename Edges>
  UncompressedGraph(Edges&& input_edges, std::size_t edge_index_offset)
      : is_exhausted_{false}, wedges_(input_edges.size()),
        num_edges_{input_edges.size()}, edge_index_offset_{edge_index_offset} {
    mpi::MPIContext ctx;
    parallel_for(0, num_edges_, [&](const std::size_t i) {
      const auto& input_edge = input_edges[i];
      WEdgeType edge;
      edge.set_src(input_edge.get_src());
      edge.set_dst(input_edge.get_dst());
      edge.set_weight(input_edge.get_weight());
      wedges_[i] = edge;
    });
  }

  WEdgeContainer get_WEdges() {
    if (is_exhausted_) {
      PRINT_WARNING("should only be called once");
    }
    is_exhausted_ = true;
    return std::move(wedges_);
  }

  WEdgeIdContainer get_WEdgeIds() const {
    if (is_exhausted_) {
      PRINT_WARNING("should only be called before get_WEdges()");
    }
    WEdgeIdContainer wedges_with_id(num_edges_);
    parallel_for(0, num_edges_, [&](const std::size_t& i) {
      auto& edge_with_id = wedges_with_id[i];
      const auto& edge = wedges_[i];
      edge_with_id.set_src(edge.get_src());
      edge_with_id.set_dst(edge.get_dst());
      edge_with_id.set_weight_and_edge_id(edge.get_weight(), edge_index_offset_ + i);
    });
    return wedges_with_id;
  }
  bool is_exhausted_; // is true if the stored edges have already been moved out
  WEdgeContainer wedges_;
  std::size_t num_edges_;
  std::size_t edge_index_offset_;
};
} // namespace hybridMST
