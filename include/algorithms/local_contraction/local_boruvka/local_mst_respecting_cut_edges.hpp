#pragma once

#include "algorithms/local_contraction/local_boruvka/mst_utils_common.hpp"
#include "algorithms/local_contraction/local_boruvka/mst_utils_cut_edges.hpp"
#include "algorithms/local_contraction/local_boruvka/mst_utils_filter.hpp"
#include "algorithms/local_contraction/local_timer.hpp"
#include "algorithms/local_contraction/utils.hpp"

#include "shared_mem_parallel.hpp"

namespace hybridMST {

//// LocalTimer lotimer().};
// template <typename EdgeType, typename MinCutWeight, typename Parents>
// std::size_t boruvka(VertexRange_ vertex_range, Span<EdgeType>& local_edges,
//                     Span<MinCutWeight> min_cut_weights, Parents& parents,
//                     GlobalEdgeId* mst) {
//
//   std::size_t m = local_edges.size();
//   const std::size_t m_initial = m;
//   std::size_t n = vertex_range.n();
//   const std::size_t n_initial = n;
//   LocalEdgeId* edge_ids = parlay_bridge::new_array_no_init<LocalEdgeId>(m);
//   LocalEdgeId* next_edge_ids =
//   parlay_bridge::new_array_no_init<LocalEdgeId>(m); VId* vertices =
//   parlay_bridge::new_array_no_init<VId>(n); VId* tmp_vertices =
//   parlay_bridge::new_array_no_init<VId>(n);
//   std::vector<std::atomic<EdgeIdWeight>> min_edges(n);
//   parlay_bridge::parallel_for(0, m, gbbs::kDefaultGranularity,
//                               [&](size_t i) { edge_ids[i] = i; });
//   parlay_bridge::parallel_for(0, n, gbbs::kDefaultGranularity, [&](size_t i)
//   {
//     vertices[i] = vertex_range.v_begin + i;
//   });
//
//   auto new_mst_edges =
//       parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
//   auto does_vertex_remain_active = parlay::sequence<bool>(n);
//   auto is_active = parlay::sequence<bool>(n);
//   size_t nb_mst_edges = 0;
//   size_t round = 0;
//   const auto normalizer = [&](const VId& v) {
//     return normalize_v(v, vertex_range);
//   };
//   mpi::MPIContext ctx;
//   // ctx.execute_in_order([&]() {
//   //   PRINT_VAR(n);
//   //   PRINT_VAR(m);
//   while (n > 0 && m > 0) {
//     gbbs_boruvka::compute_min_edges(n, m, min_edges, edge_ids, local_edges,
//                                     vertices, normalizer);
//     determine_mst_edges(n, min_edges, min_cut_weights, local_edges, vertices,
//                         parents, does_vertex_remain_active, new_mst_edges,
//                         normalizer);
//
//     nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
//         n, nb_mst_edges, new_mst_edges, local_edges, mst);
//     update_parents_min_cut_weights(n, vertices, parents, min_cut_weights,
//     params.all_edges,
//                                    normalizer);
//
//     n = gbbs_boruvka::compactify_vertices(n, vertices, tmp_vertices,
//                                           does_vertex_remain_active);
//     gbbs_boruvka::relabel_edges(m, edge_ids, local_edges, parents,
//     normalizer); m = gbbs_boruvka::filter_out_self_loops(m, edge_ids,
//     next_edge_ids); round++;
//   }
//   hybridMST::parallel_for(0, parents.size(), [&](size_t i) {
//     while (parents[i] != parents[normalizer(parents[i])]) {
//       parents[i] = parents[normalizer(parents[i])];
//     }
//   });
//
//   parlay_bridge::free_array(edge_ids, m_initial);
//   parlay_bridge::free_array(next_edge_ids, m_initial);
//   parlay_bridge::free_array(vertices, n_initial);
//   parlay_bridge::free_array(tmp_vertices, n_initial);
//   return nb_mst_edges;
// }

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
  auto print = [&](const std::string& msg) {
    if (ctx.rank() == 0) {
      // using namespace std::chrono_literals;
      // for(std::size_t i = 0; i < ctx.rank(); ++i) {
      //   std::this_thread::sleep_for(100ms);
      // }
      std::cout << "pos: " << msg << std::endl;
      std::cout << "n: " << n << " m: " << m << " initial_n: " << n_initial
                << " inital_m: " << m_initial << std::endl;
      std::cout << "\tedges: " << std::endl;
      // for (std::size_t i = 0; i < m; ++i) {
      //   std::cout << "\t" << i << " "
      //             << params.local_edges[edge_ids.get_primary_data()[i]]
      //             << " id: " << edge_ids.get_primary_data()[i] << std::endl;
      // }
      // std::cout << "\tvertices: " << std::endl;
      // for (std::size_t i = 0; i < n; ++i) {
      //   auto v = params.vertices_->get_primary_data()[i];
      //   std::cout << "\t" << i << " v: " << v
      //             << " parent: " << params.parents[v] << std::endl;
      // }
      // for (std::size_t i = 0; i < n_initial; ++i) {
      //   std::cout << "\t v: " << i << " parent: " << params.parents[i]
      //             << std::endl;
      // }
      // for (std::size_t i = 0; i < n_initial; ++i) {
      //   const auto min_cut = params.min_cut_weights[i].load();
      //   VId cut_partner = (min_cut.edge_id != LOCAL_EDGEID_UNDEFINED)
      //                         ? params.all_edges[min_cut.edge_id].get_dst()
      //                         : VID_UNDEFINED;
      //   std::cout << "\t v: " << i << " min cut: (" << min_cut.weight
      //             << ", partner: " << cut_partner << " min edge ";
      //   if (params.min_edges[i].load().edge_id < m_initial) {
      //     std::cout <<
      //     params.local_edges[params.min_edges[i].load().edge_id];
      //   }
      //   std::cout << std::endl;
      // }
    }
  };

  while (n > 0 && m > 0) {
    print("begin" + std::to_string(round));
    compute_min_edges(n, m, params.min_edges, edge_ids.get_primary_data(),
                      params.local_edges, params.vertices_->get_primary_data(),
                      EdgeIdWeightBaseComparator{}, normalizer);
    determine_mst_edges_exhausted(
        n, params.min_edges, params.min_cut_weights, params.local_edges,
        params.all_edges, params.vertices_->get_primary_data(), params.parents,
        can_be_reactivated, is_active, new_mst_edges, normalizer);

    print("before min cut update" + std::to_string(round));
    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, params.local_edges, params.mst);
    update_parents_min_cut_weights(n, params.vertices_->get_primary_data(),
                                   params.parents, params.min_cut_weights,
                                   params.all_edges, normalizer);

    num_exhausted_vertices += identify_exhausted_vertices(
        n, can_be_reactivated, params.vertices_->get_primary_data(),
        exhausted_vertices, num_exhausted_vertices);
    n = compactify_vertices(n, params.vertices_, is_active);
    relabel_edges(m, edge_ids.get_primary_data(), params.local_edges,
                  params.parents, params.min_cut_weights, normalizer);
    m = filter_out_self_loops(m, edge_ids);
    round++;
  }

  print("after loop" + std::to_string(round));
  parallel_for(0, num_exhausted_vertices, [&](const auto& i) {
    params.vertices_->get_primary_data()[i + n] = exhausted_vertices[i];
  });
  n += num_exhausted_vertices;
  hybridMST::parallel_for(0, params.n_initial, [&](size_t i) {
    while (params.parents[i] != params.parents[normalizer(params.parents[i])]) {
      params.parents[i] = params.parents[normalizer(params.parents[i])];
    }
  });

  print("finished");
  return BoruvkaResult{nb_mst_edges, n};
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

  lotimer().start("partition");
  auto [light_edges, is_balanced, pivot] =
      localMST::partition(params.local_edges);
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
  auto heavy_edges = localMST::filter(params, params.local_edges, pivot);
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
                           Span<EdgeType> local_edges, Span<EdgeType> all_edges,
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
  params.all_edges = all_edges;
  params.min_cut_weights = min_cut_weights;
  params.parents = parents;
  params.mst = mst;
  params.vertices = vertices.data();
  params.vertices_tmp_storage = vertices_tmp_storage.data();
  params.min_edges = min_edges;
  params.vertices_ = &vertices_;
  BoruvkaResult res = boruvka_filter_rec(params);
  //std::cout << "finished: " << ctx.rank() << std::endl;
  if (ctx.rank() == 0) {
    lotimer().output(std::to_string(ctx.rank()));
  }
  lotimer().reset();
  // ctx.execute_in_order([&]() { res = boruvka_filter_rec(params); });
  return res.num_mst_edges_found;
}

} // namespace hybridMST
