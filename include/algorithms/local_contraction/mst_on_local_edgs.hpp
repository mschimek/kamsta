#pragma once

#include "algorithms/local_contraction/mst_utils.hpp"
#include "algorithms/local_contraction/utils.hpp"
#include "shared_mem_parallel.hpp"

namespace hybridMST {

struct LocalTimer {
  void start(const std::string& key) {
    REORDERING_BARRIER
    start_ = std::chrono::steady_clock::now();
    REORDERING_BARRIER
    key_ = key;
  }
  void stop() {
    REORDERING_BARRIER
    auto stop_ = std::chrono::steady_clock::now();
    REORDERING_BARRIER

    std::chrono::duration<double> diff = (stop_ - start_);
    double d = diff.count();
    key_value[key_] += d;
  }
  void output(const std::string& prefix) {
    for (const auto& [key, value] : key_value) {
      std::cout << prefix << ": " << key << " " << value << std::endl;
    }
  }
  void reset() { key_value.clear(); }
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string key_;

  std::unordered_map<std::string, double> key_value;
};

LocalTimer& lotimer() {
  static LocalTimer t;
  return t;
}
// LocalTimer lotimer().};
template <typename EdgeType, typename MinCutWeight, typename Parents>
std::size_t boruvka(VertexRange_ vertex_range, Span<EdgeType>& local_edges,
                    Span<MinCutWeight> min_cut_weights, Parents& parents,
                    GlobalEdgeId* mst) {

  std::size_t m = local_edges.size();
  const std::size_t m_initial = m;
  std::size_t n = vertex_range.n();
  const std::size_t n_initial = n;
  LocalEdgeId* edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
  LocalEdgeId* next_edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
  VId* vertices = parlay_bridge::new_array_no_init<VId>(n);
  VId* tmp_vertices = parlay_bridge::new_array_no_init<VId>(n);
  std::vector<std::atomic<EdgeIdWeight>> min_edges(n);
  parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity,
                              [&](size_t i) { edge_ids[i] = i; });
  parlay_bridge::parallel_for(0, n, gbbs::kDefaultGranularity, [&](size_t i) {
    vertices[i] = vertex_range.v_begin + i;
  });

  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto does_vertex_remain_active = parlay::sequence<bool>(n);
  auto is_active = parlay::sequence<bool>(n);
  size_t nb_mst_edges = 0;
  size_t round = 0;
  const auto normalizer = [&](const VId& v) {
    return normalize_v(v, vertex_range);
  };
  mpi::MPIContext ctx;
  // ctx.execute_in_order([&]() {
  //   PRINT_VAR(n);
  //   PRINT_VAR(m);
  while (n > 0 && m > 0) {
    gbbs_boruvka::compute_min_edges(n, m, min_edges, edge_ids, local_edges,
                                    vertices, normalizer);
    determine_mst_edges(n, min_edges, min_cut_weights, local_edges, vertices,
                        parents, does_vertex_remain_active, new_mst_edges,
                        normalizer);

    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, local_edges, mst);
    update_parents_min_cut_weights(n, vertices, parents, min_cut_weights,
                                   normalizer);

    n = gbbs_boruvka::compactify_vertices(n, vertices, tmp_vertices,
                                          does_vertex_remain_active);
    gbbs_boruvka::relabel_edges(m, edge_ids, local_edges, parents, normalizer);
    m = gbbs_boruvka::filter_out_self_loops(m, edge_ids, next_edge_ids);
    round++;
  }
  hybridMST::parallel_for(0, parents.size(), [&](size_t i) {
    while (parents[i] != parents[normalizer(parents[i])]) {
      parents[i] = parents[normalizer(parents[i])];
    }
  });

  parlay_bridge::free_array(edge_ids, m_initial);
  parlay_bridge::free_array(next_edge_ids, m_initial);
  parlay_bridge::free_array(vertices, n_initial);
  parlay_bridge::free_array(tmp_vertices, n_initial);
  return nb_mst_edges;
}

template <typename Parameters> std::size_t boruvka2(Parameters& params) {
  std::size_t m = params.local_edges.size();
  const std::size_t m_initial = m;
  std::size_t n = params.vertex_range.n();
  const std::size_t n_initial = n;
  // edge id vectors need to be reallocated in each allocation as number of
  // edges might differ
  non_init_vector<LocalEdgeId> edge_ids(m);
  non_init_vector<LocalEdgeId> next_edge_ids(m);
  parallel_for(0, m, [&](size_t i) { edge_ids[i] = i; });
  parallel_for(0, n, [&](size_t i) {
    params.vertices_->get_primary_data()[i] = params.vertex_range.v_begin + i;
  });

  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto can_be_reactivated = parlay::sequence<bool>(n, false);
  auto is_active = parlay::sequence<bool>(n, true);
  size_t nb_mst_edges = 0;
  size_t round = 0;
  const auto normalizer = [&](const VId& v) {
    return normalize_v(v, params.vertex_range);
  };
  mpi::MPIContext ctx;
  // ctx.execute_in_order([&]() {
  //   PRINT_VAR(n);
  //   PRINT_VAR(m);
  while (n > 0 && m > 0) {
    std::cout << "n: " << n << " m: " << m << " initial_n: " << n_initial
              << " inital_m: " << m_initial << std::endl;
    gbbs_boruvka::compute_min_edges(
        n, m, params.min_edges, edge_ids, params.local_edges,
        params.vertices_->get_primary_data(), normalizer);
    determine_mst_edges_exhausted(
        n, params.min_edges, params.min_cut_weights, params.local_edges,
        params.vertices_->get_primary_data(), params.parents,
        can_be_reactivated, is_active, new_mst_edges, normalizer);

    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, params.local_edges, params.mst);
    update_parents_min_cut_weights(n, params.vertices_->get_primary_data(),
                                   params.parents, params.min_cut_weights,
                                   normalizer);

    n = compactify_vertices(n, params.vertices_, is_active);
    relabel_edges(m, edge_ids, params.local_edges, params.parents,
                  params.min_cut_weights, normalizer);
    auto edge_ids_data = edge_ids.data();
    auto next_edge_ids_data = next_edge_ids.data();
    m = gbbs_boruvka::filter_out_self_loops(m, edge_ids_data,
                                            next_edge_ids_data);
    std::swap(edge_ids, next_edge_ids);
    round++;
  }
  hybridMST::parallel_for(0, params.parents.size(), [&](size_t i) {
    while (params.parents[i] != params.parents[normalizer(params.parents[i])]) {
      params.parents[i] = params.parents[normalizer(params.parents[i])];
    }
  });

  std::cout << "n: " << n << " m: " << m << " initial_n: " << n_initial
            << " inital_m: " << m_initial << std::endl;
  auto vtx_range = parlay_bridge::make_slice(
      params.vertices.data() + n, params.vertices.data() + n_initial);
  n += parlay::pack_index_out(parlay_bridge::make_slice(can_be_reactivated),
                              vtx_range);

  std::cout << "n: " << n << " m: " << m << " initial_n: " << n_initial
            << " inital_m: " << m_initial << std::endl;
  return nb_mst_edges;
}

template <typename Parameters> void check(Parameters params) {
  std::size_t n = params.vertex_range.n();
  std::unordered_set<VId> set;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& v = params.vertices[i];
    set.insert(v);
    if (params.parents[v - params.vertex_range.v_begin] != v) {
      std::cout << i << " vertex " << v << " parent: "
                << params.parents[v - params.vertex_range.v_begin] << std::endl;
    }
  }
  for (const auto& e : params.local_edges) {
    const auto src = e.get_src();
    const auto dst = e.get_dst();
    if (set.find(src) == set.end() || set.find(dst) == set.end()) {
      std::cout << "endpoint not in edges: " << e << std::endl;
    }
  }
}

template <typename Parameters> BoruvkaResult boruvka3(Parameters& params) {
  // check(params);
  mpi::MPIContext ctx;

  std::size_t m = params.local_edges.size();
  const std::size_t m_initial = m;
  std::size_t n = params.vertex_range.n();
  const std::size_t n_initial = n;
  auto print = [&]() {
    if (ctx.rank() == 0) {
      std::cout << "n: " << n << " m: " << m << " initial_n: " << n_initial
                << " inital_m: " << m_initial << std::endl;
    }
  };
  // edge id vectors need to be reallocated in each allocation as number of
  // edges might differ
  SwapStorage<LocalEdgeId> edge_ids(m);
  edge_ids.initialize_primary([&](size_t i) { return i; });
  non_init_vector<VId> exhausted_vertices(n);
  std::size_t num_exhausted_vertices = 0;

  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto can_be_reactivated = parlay::sequence<bool>(n, false);
  auto is_active = parlay::sequence<bool>(n, true);
  size_t nb_mst_edges = 0;
  size_t round = 0;
  const auto normalizer = [&](const VId& v) {
    return normalize_v(v, params.vertex_range);
  };
  while (n > 0 && m > 0) {
    print();
    gbbs_boruvka::compute_min_edges(
        n, m, params.min_edges, edge_ids.get_primary_data(), params.local_edges,
        params.vertices_->get_primary_data(), normalizer);
    determine_mst_edges_exhausted(
        n, params.min_edges, params.min_cut_weights, params.local_edges,
        params.vertices_->get_primary_data(), params.parents,
        can_be_reactivated, is_active, new_mst_edges, normalizer);

    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, params.local_edges, params.mst);
    update_parents_min_cut_weights(n, params.vertices_->get_primary_data(),
                                   params.parents, params.min_cut_weights,
                                   normalizer);

    num_exhausted_vertices += identify_exhausted_vertices(
        n, can_be_reactivated, params.vertices_->get_primary_data(),
        exhausted_vertices, num_exhausted_vertices);
    n = compactify_vertices(n, params.vertices_, is_active);
    relabel_edges(m, edge_ids.get_primary_data(), params.local_edges,
                  params.parents, params.min_cut_weights, normalizer);
    m = filter_out_self_loops(m, edge_ids);
    round++;
  }

  print();
  parallel_for(0, num_exhausted_vertices, [&](const auto& i) {
    params.vertices_->get_primary_data()[i + n] = exhausted_vertices[i];
  });
  n += num_exhausted_vertices;
  hybridMST::parallel_for(0, params.n_initial, [&](size_t i) {
    while (params.parents[i] != params.parents[normalizer(params.parents[i])]) {
      params.parents[i] = params.parents[normalizer(params.parents[i])];
    }
  });

  print();
  return BoruvkaResult{nb_mst_edges, n};
}

template <typename EdgeType, typename MinCutWeight, typename Parent>
std::size_t boruvka_base(VertexRange_ vertex_range, Span<EdgeType> local_edges,
                         Span<MinCutWeight> min_cut_weights,
                         Span<Parent> parents, GlobalEdgeId* mst) {

  non_init_vector<VId> vertices(vertex_range.n());
  non_init_vector<VId> vertices_tmp_storage(vertex_range.n());
  non_init_vector<std::atomic<EdgeIdWeight>> min_edges(vertex_range.n());

  BoruvkaParameters<EdgeType, MinCutWeight, Parent, VId> params;
  params.vertex_range = vertex_range;
  params.local_edges = local_edges;
  params.min_cut_weights = min_cut_weights;
  params.parents = parents;
  params.mst = mst;
  params.vertices = vertices;
  params.vertices_tmp_storage = vertices_tmp_storage;
  params.min_edges = min_edges;
  return boruvka2(params);
}

template <typename EdgeType> Weight get_pivot(Span<EdgeType> edges) {
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
  return samples[num_samples * 0.25].get_weight();
}

bool is_unbalanced(std::size_t num_edges, std::size_t num_light_edges) {
  const std::size_t difference =
      std::abs(std::int64_t(num_edges) - std::int64_t(num_light_edges));
  const double ratio = double(difference) / num_edges;
  return std::abs(0.5 - ratio) > 0.4;
}

template <typename EdgeType> auto partition(Span<EdgeType> edges) {
  mpi::MPIContext ctx;
  const Weight pivot = get_pivot(edges);
  const auto is_light = [&](const auto& edge) {
    return edge.get_weight() <= pivot;
  };
  const auto is_heavy = [&](const auto& edge) {
    return edge.get_weight() > pivot;
  };
  if (ctx.rank() == 0) {
    std::cout << "pivot: " << pivot << std::endl;
  }
  const std::size_t num_edges = edges.size();
  const std::size_t num_light_edges = parlay::count_if(edges, is_light);

  if (is_unbalanced(num_edges, num_light_edges)) {
    return std::make_tuple(edges, edges, false);
  }

  non_init_vector<EdgeType> light_edges(num_light_edges);
  parlay::filter_into(edges, light_edges, is_light);
  non_init_vector<EdgeType> heavy_edges(edges.size() - num_light_edges);
  parlay::filter_into(edges, heavy_edges, is_heavy);
  parallel_for(0, light_edges.size(),
               [&](const auto& i) { edges[i] = light_edges[i]; });
  parallel_for(0, heavy_edges.size(), [&](const auto& i) {
    edges[i + num_light_edges] = heavy_edges[i];
  });
  Span<EdgeType> light_edges_span(edges.begin(), num_light_edges);
  Span<EdgeType> heavy_edges_span(edges.begin() + num_light_edges,
                                  heavy_edges.size());
  return std::make_tuple(light_edges_span, heavy_edges_span, true);
}

template <typename EdgeType, typename MinCutEdges, typename Normalizer>
auto partition(Span<EdgeType> edges, const MinCutEdges& min_cut_weights,
               const Normalizer& normalizer) {
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
  const Weight pivot = get_pivot(edges);
  lotimer().stop();
  const auto is_light = [&](const auto& edge) {
    return edge.get_weight() <= pivot;
  };
  const auto is_heavy = [&](const auto& edge) {
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
    return std::make_tuple(edges, edges, false, Weight{0});
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
  return std::make_tuple(std::move(light_edges), heavy_edges_span, true, pivot);
}

template <typename Edges> auto partition2(Edges& edges) {
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
  const Weight pivot = get_pivot(edges);
  lotimer().stop();
  const auto is_light = [&](const auto& edge) {
    return edge.get_weight() <= pivot;
  };
  const auto is_heavy = [&](const auto& edge) {
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
  const std::size_t n = params.n_initial;
  const std::size_t n_prime = params.vertex_range.n();
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

template <typename Params> auto boruvka_filter_rec(Params params) {
  mpi::MPIContext ctx;
  if (params.vertex_range.n() <= 1) {
    return BoruvkaResult{0, params.vertex_range.n()};
    // return 0ul;
  }
  if (params.local_edges.size() <= 3 * params.vertex_range.n()) {
    lotimer().start("boruvka");
    const auto res = boruvka3(params);
    lotimer().stop();
    return res;
  }

  const auto normalizer = [&](const VId& v) {
    return normalize_v(v, params.vertex_range);
  };

  lotimer().start("partition");
  auto [light_edges, is_balanced, pivot] =
      partition2(params.local_edges);
  lotimer().stop();
  if (!is_balanced) {
    lotimer().start("boruvka");
    const auto res = boruvka3(params);
    lotimer().stop();
    return res;
  }
  // PRINT_CONTAINER_WITH_INDEX(light_edges);
  // PRINT_CONTAINER_WITH_INDEX(heavy_edges);

  Params light_params = params;
  light_params.local_edges = Span(light_edges);
  auto num_light_edges = light_edges.size();
  auto num_heavy_edges = params.local_edges.size() - num_light_edges;
  const auto [num_mst_edges, num_active_vertices] =
      boruvka_filter_rec(light_params);
  lotimer().start("filter");
  auto heavy_edges = filter(params, params.local_edges, pivot);
  lotimer().stop();
  if (ctx.rank() == 0) {
    std::cout << "light edges: " << num_light_edges
              << " heavy edges: " << num_heavy_edges
              << " filtered edges: " << heavy_edges.size()
              << " active vertices: " << num_active_vertices << std::endl;
  }
  Params heavy_params = params;
  heavy_params.mst += num_mst_edges;
  heavy_params.local_edges = heavy_edges;
  heavy_params.vertex_range.v_end =
      heavy_params.vertex_range.v_begin + num_active_vertices;
  lotimer().start("boruvka");
  auto res = boruvka3(heavy_params);
  lotimer().stop();
  res.num_mst_edges_found += num_mst_edges;
  return res;
}

template <typename EdgeType, typename MinCutWeight, typename Parent>
std::size_t boruvka_filter(VertexRange_ vertex_range,
                           Span<EdgeType> local_edges,
                           Span<MinCutWeight> min_cut_weights,
                           Span<Parent> parents, GlobalEdgeId* mst) {

  mpi::MPIContext ctx;

  const auto n = vertex_range.n();
  non_init_vector<VId> vertices(n);
  non_init_vector<VId> vertices_tmp_storage(n);
  non_init_vector<std::atomic<EdgeIdWeight>> min_edges(n);
  parallel_for(0, vertices.size(),
               [&](size_t i) { vertices[i] = vertex_range.v_begin + i; });

  SwapStorage<VId> vertices_(n);

  parallel_for(0, n, [&](size_t i) {
    vertices_.get_primary_data()[i] = vertex_range.v_begin + i;
  });
  BoruvkaParameters<EdgeType, MinCutWeight, Parent, VId> params;
  params.n_initial = vertex_range.n();
  params.vertex_range = vertex_range;
  params.local_edges = local_edges;
  params.min_cut_weights = min_cut_weights;
  params.parents = parents;
  params.mst = mst;
  params.vertices = vertices.data();
  params.vertices_tmp_storage = vertices_tmp_storage.data();
  params.min_edges = min_edges;
  params.vertices_ = &vertices_;
  BoruvkaResult res = boruvka_filter_rec(params);
  if(ctx.rank() == 0) {
  lotimer().output(std::to_string(ctx.rank()));
  }
  lotimer().reset();
  // ctx.execute_in_order([&]() { res = boruvka_filter_rec(params); });
  return res.num_mst_edges_found;
}
} // namespace hybridMST
