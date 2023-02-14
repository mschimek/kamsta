#pragma once
#include <execution>

#include "algorithms/hybrid_boruvka_substeps/is_representative_computation.hpp"
#include "definitions.hpp"
#include "mpi/context.hpp"
#include "util/utils.hpp"

namespace hybridMST {
struct EdgeRenamer {
  template <typename Graph, typename LookupMap, typename Container>
  static void
  rename_edges(Graph& graph, const Container& new_names_local_vertices,
               // absl::flat_hash_map<VId, VId>& new_names_ghost_vertices) {
               const LookupMap& new_names_ghost_vertices) {
    mpi::MPIContext ctx;
    const std::size_t nb_edges = graph.edges().size();
    const auto& locator = graph.locator();
    // auto handles = growt::create_handle_ets(new_names_ghost_vertices);
#pragma omp parallel for
    for (std::size_t i = 0; i < nb_edges; ++i) {
      auto& edge = graph.edges()[i];
      auto old_edge = edge;
      VId src = edge.get_src();
      VId dst = edge.get_dst();
      const VId src_local = graph.get_local_id(src);
      src = new_names_local_vertices[src_local];
      edge.set_src(src);
      if (locator.is_local(dst)) {
        const VId dst_local = graph.get_local_id(dst);
        dst = new_names_local_vertices[dst_local];
        edge.set_dst(dst);
      } else {
        auto it = new_names_ghost_vertices.find(dst + 1);
        // should only point to end if a parallel edge is split between two
        // processor in this case the edge with the higher weight is not needed
        dst = (it == new_names_ghost_vertices.end()) ? src : (*it).second;
        edge.set_dst(dst);
      }
    }
  }
};

struct AddMstEdgesSeq {
  template <typename Graph>
  static void execute(const Graph& graph,
                      const non_init_vector<std::uint8_t>& is_rep,
                      const non_init_vector<LocalEdgeId>& min_edges_idxs,
                      non_init_vector<GlobalEdgeId>& mst_edges) {
    for (std::size_t i = 0; i < is_rep.size(); ++i) {
      const auto& edge_id = min_edges_idxs[i];
      if (is_rep[i] == 1 || !is_defined(edge_id))
        continue;
      const auto& edge = graph.edges()[edge_id];
      mst_edges.push_back(edge.global_id);
    }
  }
  template <typename Graph>
  static void execute(const Graph& graph, Representatives& representatives,
                      const non_init_vector<LocalEdgeId>& min_edges_idxs,
                      non_init_vector<GlobalEdgeId>& mst_edges) {
    for (std::size_t i = 0; i < graph.local_n(); ++i) {
      const auto& edge_id = min_edges_idxs[i];
      if (representatives.is_root(i) || !is_defined(edge_id))
        continue;
      const auto& edge = graph.edges()[edge_id];
      mst_edges.push_back(edge.get_edge_id());
    }
  }
};

struct AddMstEdgesPreAlloc {
  template <typename Graph>
  static Span<GlobalEdgeId>
  execute(const Graph& graph, const non_init_vector<std::uint8_t>& is_rep,
          const non_init_vector<LocalEdgeId>& min_edges_idxs,
          Span<GlobalEdgeId> mst_edges) {
    std::size_t num_added_mst_ids = 0;
    for (std::size_t i = 0; i < is_rep.size(); ++i) {
      const auto& edge_id = min_edges_idxs[i];
      if (is_rep[i] == 1 || !is_defined(edge_id))
        continue;
      const auto& edge = graph.edges()[edge_id];
      mst_edges[num_added_mst_ids++] = edge.get_edge_id();
    }
    return Span<GlobalEdgeId>(mst_edges.data() + num_added_mst_ids,
                              mst_edges.size() - num_added_mst_ids);
  }
};

} // namespace hybridMST
