#pragma once

#include <random>

#include "gbbs_bridge/bridge.hpp"

#include "algorithms/local_contraction/local_timer.hpp"
#include "definitions.hpp"
#include "util/utils.hpp"

namespace hybridMST {
namespace localMST {
template <typename EdgeType> Weight get_pivot(Span<EdgeType> edges, double balance_ratio = 0.25) {
  if (edges.size() == 0) {
    return 1;
  }
  if (edges.size() <= 2) {
    return edges[0].get_weight();
  }
  // PRINT_CONTAINER_WITH_INDEX(edges);
  std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, edges.size() - 1);
  const std::size_t num_samples = 50;
  std::vector<EdgeType> samples(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    samples[i] = edges[distrib(gen)];
  }
  std::sort(samples.begin(), samples.end(), WeightOrder<EdgeType>{});
  return samples[num_samples * balance_ratio].get_weight();
}

inline bool is_unbalanced(std::size_t num_edges, std::size_t num_light_edges) {
  const std::size_t difference =
      std::abs(std::int64_t(num_edges) - std::int64_t(num_light_edges));
  const double ratio = double(difference) / num_edges;
  return std::abs(0.5 - ratio) > 0.4;
}

template <typename Edges> auto partition(Edges& edges, double balance_ratio = 0.25) {
  using EdgeType = typename Edges::value_type;
  lotimer().start("min_weight");
  mpi::MPIContext ctx;
  // is too slow
  // const auto is_heavier_than_cut_edges = [&](const auto& edge) {
  //  const auto w = edge.get_weight();
  //  const bool is_heavier_than_cut_edge_src =
  //      min_cut_weights[normalizer(edge.get_src())].load().weight < w;
  //  const bool is_heavier_than_cut_edge_dst =
  //      min_cut_weights[normalizer(edge.get_dst())].load().weight < w;
  //  return !is_heavier_than_cut_edge_src | !is_heavier_than_cut_edge_dst;
  //};

  //{
  //  const auto tmp = parlay::filter(edges, is_heavier_than_cut_edges);
  //  parallel_for(0, tmp.size(), [&](const auto& i) { edges[i] = tmp[i]; });
  //}

  lotimer().stop();
  lotimer().start("pivot");
  const Weight pivot = get_pivot(edges, balance_ratio);
  lotimer().stop();
  const auto is_light = [&](const auto& edge) {
    return edge.get_weight() <= pivot;
  };
  [[maybe_unused]] const auto is_heavy = [&](const auto& edge) {
    return edge.get_weight() > pivot;
  };
  if (ctx.rank() == 0) {
    std::cout << "pivot: " << pivot << std::endl;
  }
  lotimer().start("unbalanced_check");
  const std::size_t num_edges = edges.size();
  const std::size_t num_light_edges = parlay::count_if(edges, is_light);
  lotimer().stop();

  if (is_unbalanced(num_edges, num_light_edges)) {
    return std::make_tuple(non_init_vector<EdgeType>{}, false, Weight{0});
  }

  lotimer().start("filter__");
  non_init_vector<EdgeType> light_edges(num_light_edges);
  parlay::filter_into(edges, light_edges, is_light);
  // non_init_vector<EdgeType> heavy_edges(edges.size() - num_light_edges);
  // parlay::filter_into(edges, heavy_edges, is_heavy);
  lotimer().stop();

  lotimer().start("assign");
  // parallel_for(0, light_edges.size(),
  //              [&](const auto& i) { edges[i] = light_edges[i]; });
  // parallel_for(0, heavy_edges.size(), [&](const auto& i) {
  //   edges[i + num_light_edges] = heavy_edges[i];
  // });
  lotimer().stop();
  Span<EdgeType> light_edges_span(edges.begin(), num_light_edges);
  // Span<EdgeType> heavy_edges_span(edges.begin() + num_light_edges,
  //                                 heavy_edges.size());
  Span<EdgeType> heavy_edges_span(edges.begin(), edges.size());
  return std::make_tuple(std::move(light_edges), true, pivot);
}

// todo adapt to active vertices
template <typename Params, typename EdgeType>
Span<EdgeType> filter(Params& params, Span<EdgeType> heavy_edges,
                      Weight pivot) {
  const auto normalizer = [&](const VId& v) {
    return normalize_v(v, params.vertex_range);
  };
  parallel_for(0, heavy_edges.size(), [&](size_t i) {
    auto& e = heavy_edges[i];
    const VId u = normalizer(e.get_src());
    const VId v = normalizer(e.get_dst());
    const bool is_smaller_pivot =
        e.get_weight() <= pivot; // will be filtered out
    e.set_src((1 - is_smaller_pivot) * params.parents[u]);
    e.set_dst((1 - is_smaller_pivot) * params.parents[v]);
  });
  auto is_edge_not_self_loop = [&](const auto& edge) {
    return edge.get_dst() != edge.get_src();
  };
  auto edges = parlay::filter(
      parlay_bridge::make_slice(heavy_edges.data(), heavy_edges.size()),
      is_edge_not_self_loop);
  parallel_for(0, edges.size(),
               [&](const auto& i) { heavy_edges[i] = edges[i]; });
  return Span<EdgeType>(heavy_edges.data(), edges.size());
}
} // namespace localMST
} // namespace hybridMST
