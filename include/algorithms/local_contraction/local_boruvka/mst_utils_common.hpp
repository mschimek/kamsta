#pragma once

#include "algorithms/local_contraction/utils.hpp"
#include "definitions.hpp"
#include "shared_mem_parallel.hpp"
#include "util/atomic_ops.hpp"
#include "util/utils.hpp"

namespace hybridMST {

struct BoruvkaResult {
  std::size_t num_mst_edges_found;
  std::size_t num_active_vertices;
};

template <typename T> struct SwapStorage {
  SwapStorage(std::size_t n) : primary_data(n), secondary_data(n) {}
  template <typename F> void initialize_primary(F&& f) {
    parallel_for(0, primary_data.size(),
                 [&](size_t i) { primary_data[i] = f(i); });
  }
  void swap() { std::swap(primary_data, secondary_data); }
  T* get_primary_data() { return primary_data.data(); }
  T* get_secondary_data() { return secondary_data.data(); }
  non_init_vector<T> primary_data;
  non_init_vector<T> secondary_data;
};

/// Boruvka Parameters needed for local contraction procedure with min cut edges
template <typename EdgeType, typename MinCutWeight, typename Parent,
          typename Vertex>
struct BoruvkaParameters {
  std::size_t n_initial;
  VertexRange_ vertex_range;  // as vertex ranges are [v_begin, v_end) with
                              // v_begin >= 0, we have to pass v_bgin, too
  Span<EdgeType> local_edges; // non owning pointer to the local edges
  Span<EdgeType> all_edges;   // non owning pointer to all edges
  Span<MinCutWeight> min_cut_weights; // non owning pointer to an array
                                      // containing the minimum edge weight of a
                                      // cut edge (v,u) for every local vertex v
  Span<Parent> parents;               // implicitly stores the MST found so far
  GlobalEdgeId* mst; // stores the MST edges (via their global edge ids)
  Vertex*
      vertices; // contains the vertex ids that are still relevant (needed as we
                // do not compactify the vertices after a boruvka round)
  Vertex* vertices_tmp_storage; // tempoary storage needed to filter vertices
  SwapStorage<VId>* vertices_;  //
  Span<std::atomic<EdgeIdWeight>> min_edges;
};

/// Boruvka Parameters needed for plain boruvka
template <typename EdgeType, typename Parent, typename Vertex>
struct PlainBoruvkaParameters {
  std::size_t n_initial;
  VertexRange_ vertex_range; // as vertex ranges are [v_begin, v_end) with
                             // v_begin >= 0, we have to pass v_bgin, too
  Span<EdgeType> edges;      // non owning pointer to the local edges
  Span<Parent> parents;      // implicitly stores the MST found so far
  GlobalEdgeId* mst;         // stores the MST edges (via their global edge ids)
  // Vertex* vertices;          // contains the vertex ids that are still
  // relevant
  //  (needed as we do not compactify the vertices after a
  //  boruvka round)
  Vertex* vertices_tmp_storage; // tempoary storage needed to filter vertices
  SwapStorage<VId>* vertices_;  //
  Span<std::atomic<EdgeIdWeight>> min_edges;
  friend std::ostream& operator<<(std::ostream& out,
                                  const PlainBoruvkaParameters& param) {
    return out << "(n_initial: " << param.n_initial
               << " edges: " << param.edges.size() << " vertex_range: ("
               << param.vertex_range.v_begin << ", " << param.vertex_range.v_end
               << ")";
  }
};

struct EdgeIdWeightBaseComparator {
  bool operator()(const EdgeIdWeight& lhs, const EdgeIdWeight& rhs) const {
    return ((lhs.weight < rhs.weight) ||
            ((lhs.weight == rhs.weight) && (lhs.edge_id < rhs.edge_id)));
  }
};
/// The edges which are considered for this selection process are the ones whose
/// ids are stored in edge_ids. Take global EdgeIds into account
template <typename Edges, typename MinEdges, typename EdgeIds,
          typename Vertices, typename EdgeIdWeightComp,
          typename VertexNormalization = Identity>
void compute_min_edges(
    std::size_t n, std::size_t m, MinEdges& min_edges, const EdgeIds& edge_ids,
    const Edges& edges, const Vertices& vertices,
    const EdgeIdWeightComp& edgeIdWeight_comp,
    VertexNormalization&& normalizer = VertexNormalization{}) {
  hybridMST::parallel_for(0, n, [&](size_t i) {
    VId v = vertices[i];
    min_edges[normalizer(v)] = EdgeIdWeight{LOCAL_EDGEID_UNDEFINED, WEIGHT_MAX};
  });
  hybridMST::parallel_for(0, m, [&](size_t i) {
    const LocalEdgeId e_id = edge_ids[i];
    const auto& edge = edges[e_id];
    const EdgeIdWeight id_weight{e_id, edge.get_weight()};
    hybridMST::write_min_gbbs(min_edges[normalizer(edge.get_src())], id_weight,
                              edgeIdWeight_comp);
    hybridMST::write_min_gbbs(min_edges[normalizer(edge.get_dst())], id_weight,
                              edgeIdWeight_comp);
  });
}
template <typename IsExhausted, typename Vertices>
std::size_t
identify_exhausted_vertices(std::size_t n, IsExhausted& is_exhausted,
                            const Vertices& vertices,
                            non_init_vector<VId>& exhausted_vertices,
                            std::size_t num_exhausted_vertices) {

  auto in = parlay::slice(vertices, vertices + n);
  auto out = parlay::slice(exhausted_vertices.begin() + num_exhausted_vertices,
                           exhausted_vertices.end());
  const auto num_new_exhausted_vertices =
      parlay::pack_into(in, is_exhausted, out);
  parallel_for(0, n, [&](const auto& i) { is_exhausted[i] = false; });
  return num_new_exhausted_vertices;
}

template <typename RootInfo>
std::size_t compactify_vertices(std::size_t n, SwapStorage<VId>* vertices,
                                const RootInfo& is_root) {
  const auto ptr_primary = vertices->get_primary_data();
  const auto ptr_secondary = vertices->get_secondary_data();
  auto vertices_input = parlay::slice(ptr_primary, ptr_primary + n);
  auto vertices_output = parlay::slice(ptr_secondary, ptr_secondary + n);
  n = parlay::pack_out(vertices_input, is_root, vertices_output);
  vertices->swap();
  return n;
}

inline std::size_t filter_out_self_loops_(std::size_t m,
                                          SwapStorage<LocalEdgeId>& edge_ids) {
  auto is_not_self_loop = [&](size_t i) {
    const bool res = !(edge_ids.get_primary_data()[i] & MSD<LocalEdgeId>);
    // std::cout << " filter process: " << i << " " <<
    // edge_ids.get_primary_data()[i] << " res: " << res << std::endl;
    return res;
  };
  auto self_loop_input = parlay::delayed_seq<bool>(m, is_not_self_loop);
  const auto ptr_in = edge_ids.get_primary_data();
  const auto ptr_out = edge_ids.get_secondary_data();
  auto edge_id_input_slice = parlay::slice(ptr_in, ptr_in + m);
  auto edge_id_output_slice = parlay::slice(ptr_out, ptr_out + m);

  m = parlay::pack_out(edge_id_input_slice, self_loop_input,
                       edge_id_output_slice);
  edge_ids.swap();
  // for(std::size_t i = 0; i < m; ++i) {
  //   std::cout << "\t surviving ids: " << i  << ": " <<
  //   edge_ids.get_primary_data()[i] << std::endl;
  // }
  return m;
}
inline std::size_t filter_out_self_loops(std::size_t m,
                                         SwapStorage<LocalEdgeId>& edge_ids) {
  auto is_not_self_loop = [&](size_t i) {
    const bool res = !(edge_ids.get_primary_data()[i] & MSD<LocalEdgeId>);
    // std::cout << " filter process: " << i << " " <<
    // edge_ids.get_primary_data()[i] << " res: " << res << std::endl;
    return res;
  };
  auto self_loop_input = parlay::delayed_seq<bool>(m, is_not_self_loop);
  const auto ptr_in = edge_ids.get_primary_data();
  const auto ptr_out = edge_ids.get_secondary_data();
  auto edge_id_input_slice = parlay::slice(ptr_in, ptr_in + m);
  auto edge_id_output_slice = parlay::slice(ptr_out, ptr_out + m);

  m = parlay::pack_out(edge_id_input_slice, self_loop_input,
                       edge_id_output_slice);
  edge_ids.swap();
  return m;
}

} // namespace hybridMST
