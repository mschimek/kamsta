#pragma once

namespace hybridMST {

/// shifts every vertex range [v_begin, v_end) to [0, v_end - v_begin)
inline VId normalize_v(const VId& v, const VertexRange_ range) {
  return v - range.v_begin;
}
/// a vertex is true local if it cannot be split over multiple PEs,
/// i.e. not the first and not the last one.
inline bool is_true_local(const VId& v, const VertexRange_ range) {
  return ((range.v_begin < v) && ((v + 1) < range.v_end));
}

template <typename Edge>
inline bool is_true_local(const Edge& e, const VertexRange_ range) {
  return is_true_local(e.get_src(), range) && is_true_local(e.get_dst(), range);
}

template <typename Edges>
inline bool has_degree_one(std::size_t index, const Edges& edges) {
  if (index == 0 || index + 1 == edges.size())
    return false;
  const auto& cur_src = edges[index].get_src();
  const auto& prev_edge = edges[index - 1];
  const auto& next_edge = edges[index + 1];
  return (cur_src != prev_edge.get_src()) && (cur_src != next_edge.get_src());
}

template <typename Edges>
auto get_local_edges(const Edges& edges, const VertexRange_ range) {
  using EdgeType = typename Edges::value_type;
  auto local_edges = parlay::filter(edges, [&](const auto& edge) {
    return (edge.get_src() < edge.get_dst()) && is_true_local(edge, range);
  });
  return local_edges;
}

template <typename Edges, typename MinCutWeights>
auto get_local_edges(const Edges& edges, const VertexRange_ range, const MinCutWeights& min_cut_weights) {
  using EdgeType = typename Edges::value_type;
  auto normalizer = [&range](const VId& v) {
    return normalize_v(v, range);
  };
  auto local_edges = parlay::filter(edges, [&](const auto& edge) {
    if(!is_true_local(edge, range) || (edge.get_src() > edge.get_dst())) {
      return false;
    }
      const auto w = edge.get_weight();
      const bool is_heavier_than_cut_edge_src =
        min_cut_weights[normalizer(edge.get_src())].load().weight < w;
      const bool is_heavier_than_cut_edge_dst =
        min_cut_weights[normalizer(edge.get_dst())].load().weight < w;
    const bool is_not_heavier_than_cut_edges = !is_heavier_than_cut_edge_src | !is_heavier_than_cut_edge_dst;
    return is_not_heavier_than_cut_edges;
  });
  return local_edges;
}

} // namespace hybridMST
