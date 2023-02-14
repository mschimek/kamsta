#pragma once

#include "algorithms/local_contraction/utils.hpp"
#include "shared_mem_parallel.hpp"

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

template <typename EdgeType, typename MinCutWeight, typename Parent,
          typename Vertex>
struct BoruvkaParameters {
  std::size_t n_initial;
  VertexRange_ vertex_range;  // as vertex ranges are [v_begin, v_end) with
                              // v_begin >= 0, we have to pass v_bgin, too
  Span<EdgeType> local_edges; // non owning pointer to the local edges
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

template <typename EdgeIds, typename Edges, typename Parents,
          typename MinCutWeights, typename VertexNormalization = Identity>
void relabel_edges(std::size_t m, const EdgeIds& edge_ids, Edges& edges,
                   const Parents& parents, const MinCutWeights& min_cut_weights,
                   VertexNormalization&& normalizer = VertexNormalization{}) {
  parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity, [&](size_t i) {
    size_t e_id = edge_ids[i];
    auto& edge = edges[e_id];
    const VId u = static_cast<VId>(edge.get_src());
    const VId v = static_cast<VId>(edge.get_dst());
    const auto w = edge.get_weight();
    const VId pu = parents[normalizer(u)];
    const VId pv = parents[normalizer(v)];
    if (u != pu || v != pv) {
      edge.set_src(pu);
      edge.set_dst(pv);
    }
    const bool is_heavier_than_cut_edge_pu =
        min_cut_weights[normalizer(pu)].load().weight < w;
    const bool is_heavier_than_cut_edge_pv =
        min_cut_weights[normalizer(pv)].load().weight < w;
    const bool is_heavier_than_cut_edges =
        is_heavier_than_cut_edge_pu & is_heavier_than_cut_edge_pv;

    if (pu == pv || is_heavier_than_cut_edges) {
      edge_ids[i] |=
          MSD<LocalEdgeId>; // mark self loops, so that they can be filtered out
    }
  });
}

template <typename MinEdges, typename MinCutWeights, typename Edges,
          typename Vertices, typename Parents, typename RemainsActive,
          typename MstEdges, typename VertexNormalizer>
void determine_mst_edges(std::size_t n, const MinEdges& min_edges,
                         const MinCutWeights& min_cut_weights,
                         const Edges& edges, const Vertices& vertices,
                         Parents& parents, RemainsActive& remains_active,
                         MstEdges& new_mst_edges,
                         VertexNormalizer&& normalizer) {
  using namespace parlay_bridge;
  mpi::MPIContext ctx;
  // ctx.execute_in_order([&](){
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const VId normalized_v = normalizer(v);
    const EdgeIdWeight& e = min_edges[normalized_v].load();
    if (e.edge_id == LOCAL_EDGEID_UNDEFINED) {
      // no more local edges incident to v
      remains_active[i] = false;
      new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
    } else {
      const LocalEdgeId min_edge_idx = e.edge_id;
      const auto& min_edge = edges[min_edge_idx];
      const VId u = static_cast<VId>(min_edge.get_src()) ^
                    static_cast<VId>(min_edge.get_dst()) ^ v;
      const VId normalized_u = normalizer(u);
      if (min_edge.get_weight() > min_cut_weights[normalized_v].load().weight) {
        // min local edge heavier than min cut edge -> v cannot be hooked on
        // another vertex
        remains_active[i] =
            false; // v can still be the "root" but new vertices hooked on v
                   // cannot add new local edges to v that are lighter than its
                   // minimal cut edge.
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else if (min_edge.get_weight() >
                 min_cut_weights[normalized_u].load().weight) {
        // min local edge heavier than u's min cut edge, but lighter/equal to
        // v's min cut edge -> v must be hooked on u
        parents[normalized_v] = u;
        remains_active[i] = false;
        new_mst_edges[i] = min_edge_idx;
      }
      // non-local edges are not important -> normal boruvka
      else if (u > v &&
               min_edge_idx == min_edges[normalized_u].load().edge_id) {
        // break ties in case of pseudo-case root (lower endpoint is chosen)
        parents[normalized_v] = v;
        remains_active[i] = true;
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else {
        // normal case: min cut edges of both endpoints heavier than local min
        // edge and no pseudo tree roots
        parents[normalized_v] = u;
        remains_active[i] = false;
        new_mst_edges[i] = min_edge_idx;
      }
    }
  });
}

// changed
template <typename MinEdges, typename MinCutWeights, typename Edges,
          typename Vertices, typename Parents, typename CanBeReactivated,
          typename RemainsActive, typename MstEdges, typename VertexNormalizer>
void determine_mst_edges_exhausted(std::size_t n, const MinEdges& min_edges,
                                   const MinCutWeights& min_cut_weights,
                                   const Edges& edges, const Vertices& vertices,
                                   Parents& parents,
                                   CanBeReactivated& can_be_reactivated,
                                   RemainsActive& remains_active,
                                   MstEdges& new_mst_edges,
                                   VertexNormalizer&& normalizer) {
  mpi::MPIContext ctx;
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const VId normalized_v = normalizer(v);
    const EdgeIdWeight& e = min_edges[normalized_v].load();
    if (e.edge_id == LOCAL_EDGEID_UNDEFINED) {
      //  no more local edges incident to v
      remains_active[i] = false;
      can_be_reactivated[i] = true;
      new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
    } else {
      const LocalEdgeId min_edge_idx = e.edge_id;
      const auto& min_edge = edges[min_edge_idx];
      const VId u = static_cast<VId>(min_edge.get_src()) ^
                    static_cast<VId>(min_edge.get_dst()) ^ v;
      const VId normalized_u = normalizer(u);
      if (min_edge.get_weight() > min_cut_weights[normalized_v].load().weight) {
        // min local edge heavier than min cut edge -> v cannot be hooked on
        // another vertex
        remains_active[i] =
            false; // v can still be the "root" but new vertices hooked on v
                   // cannot add new local edges to v that are lighter than its
                   // minimal cut edge.
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else if (min_edge.get_weight() >
                 min_cut_weights[normalized_u].load().weight) {
        // min local edge heavier than u's min cut edge, but lighter/equal to
        // v's min cut edge -> v must be hooked on u
        parents[normalized_v] = u;
        remains_active[i] = false;
        new_mst_edges[i] = min_edge_idx;
      }
      // non-local edges are not important -> normal boruvka
      else if (u > v &&
               min_edge_idx == min_edges[normalized_u].load().edge_id) {
        // break ties in case of pseudo-case root (lower endpoint is chosen)
        parents[normalized_v] = v;
        remains_active[i] = true;
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else {
        // normal case: min cut edges of both endpoints heavier than local min
        // edge and no pseudo tree roots
        parents[normalized_v] = u;
        remains_active[i] = false;
        new_mst_edges[i] = min_edge_idx;
      }
    }
  });
}

template <typename Vertices, typename Parents, typename MinCutWeights,
          typename VertexNormalizer>
void update_parents_min_cut_weights(std::size_t n, const Vertices& vertices,
                                    Parents& parents,
                                    MinCutWeights& min_cut_weights,
                                    VertexNormalizer&& normalizer) {

  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const VId normalized_v = normalizer(v);
    while (parents[normalized_v] !=
           parents[normalizer(parents[normalized_v])]) {
      parents[normalized_v] = parents[normalizer(parents[normalized_v])];
    }
  });
  const auto comp = [](const EdgeIdWeight& lhs, const EdgeIdWeight& rhs) {
    return std::tie(lhs.weight, lhs.edge_id) <
           std::tie(rhs.weight, rhs.edge_id);
  };
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const VId normalized_v = normalizer(v);
    const VId parent_v = parents[normalized_v];
    const VId normalized_parent_v = normalizer(parent_v);
    write_min_gbbs(min_cut_weights[normalized_parent_v],
                   min_cut_weights[normalized_v].load(), comp);
  });
}

template <typename EdgeType, typename IsLocalEdge>
non_init_vector<std::atomic<EdgeIdWeight>>
compute_min_cut_weights(const Span<EdgeType> edges,
                        const VertexRange_ vertex_range,
                        IsLocalEdge&& is_local_edge) {
  const auto comp = [](const EdgeIdWeight& lhs, const EdgeIdWeight& rhs) {
    return std::tie(lhs.weight, lhs.edge_id) <
           std::tie(rhs.weight, rhs.edge_id);
  };
  const std::size_t n = vertex_range.n();
  non_init_vector<std::atomic<EdgeIdWeight>> min_cut_weights(n);
  hybridMST::parallel_for(0, n, [&](size_t i) {
    min_cut_weights[i] = EdgeIdWeight{LOCAL_EDGEID_UNDEFINED, WEIGHT_MAX};
  });
  auto normalize = [&vertex_range](const VId& v) {
    return normalize_v(v, vertex_range);
  };
  hybridMST::parallel_for(0, edges.size(), [&](std::size_t i) {
    const auto& edge = edges[i];
    if (is_local_edge(edge))
      return;
    const std::size_t normalized_src = normalize(edge.get_src());
    const LocalEdgeId e_id = i;
    const EdgeIdWeight id_weight{e_id, edge.get_weight()};
    hybridMST::write_min_gbbs(min_cut_weights[normalize(edge.get_src())],
                              id_weight, comp);
  });
  return min_cut_weights;
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

std::size_t filter_out_self_loops(std::size_t m,
                                  SwapStorage<LocalEdgeId>& edge_ids) {
  auto is_not_self_loop = [&](size_t i) {
    return !(edge_ids.get_primary_data()[i] & MSD<LocalEdgeId>);
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
