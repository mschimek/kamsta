#pragma once

#include "algorithms/gbbs_reimplementation.hpp"
#include "algorithms/local_contraction/local_boruvka/mst_utils_common.hpp"
#include "algorithms/local_contraction/local_boruvka/mst_utils_filter.hpp"
#include "algorithms/local_contraction/local_boruvka/mst_utils_plain.hpp"
#include "definitions.hpp"
#include "shared_mem_parallel.hpp"
#include "util/utils.hpp"

namespace hybridMST {
namespace recursive_plain_filter_boruvka {

template <typename Parameters> BoruvkaResult boruvka(Parameters& params) {
  // check(params);
  mpi::MPIContext ctx;

  std::size_t m = params.edges.size();
  const std::size_t m_initial = m;
  std::size_t n = params.vertex_range.n();
  const std::size_t n_initial = n;
  SwapStorage<LocalEdgeId> edge_ids(m);

  // edge id vectors need to be reallocated in each allocation as number of
  // edges might differ
  edge_ids.initialize_primary([&](size_t i) { return i; });
  non_init_vector<VId> exhausted_vertices(n);
  std::size_t num_exhausted_vertices = 0;

  auto new_mst_edges =
      parlay::sequence<GlobalEdgeId>(n, LOCAL_EDGEID_UNDEFINED);
  auto is_exhausted = parlay::sequence<bool>(n, false);
  auto is_root = parlay::sequence<bool>(n, true);
  size_t nb_mst_edges = 0;
  size_t round = 0;
  const auto normalizer = [&](const VId& v) { return v; };
  auto print = [&](std::string pos) {
    if (ctx.rank() == 0) {
      std::cout << "pos: " << pos << std::endl;
      std::cout << "\tn: " << n << " m: " << m << " initial_n: " << n_initial
                << " inital_m: " << m_initial << std::endl;
    }
    // std::cout << "\tedges: " << std::endl;
    // for(std::size_t i = 0; i < m; ++i) {
    //   std::cout << "\t" << i << " " <<
    //   params.edges[edge_ids.get_primary_data()[i]] << " id: " <<
    //   edge_ids.get_primary_data()[i] << std::endl;
    // }
    // std::cout << "\tvertices: " << std::endl;
    // for(std::size_t i = 0; i < n; ++i) {
    //   auto v = params.vertices_->get_primary_data()[i];
    //   std::cout << "\t" << i << " v: " << v << " parent: " <<
    //   params.parents[v] << std::endl;
    // }
    // for(std::size_t i = 0; i < n_initial; ++i) {
    //   std::cout << "\t v: " << i << " parent: " << params.parents[i] << " is
    //   exhausted: " << is_exhausted[i] << std::endl;
    // }
    // std::cout << "exhausted vertices: " << std::endl;
    // for(std::size_t i = 0; i < num_exhausted_vertices; ++i) {
    //   std::cout << "\t  " << exhausted_vertices[i] << std::endl;
    // }
  };
  while (n > 0 && m > 0) {
    print("begin " + std::to_string(round));
    compute_min_edges(n, m, params.min_edges, edge_ids.get_primary_data(),
                      params.edges, params.vertices_->get_primary_data(),
                      EdgeIdWeightBaseComparator{}, normalizer);
    determine_mst_edges(n, params.min_edges, params.edges,
                        params.vertices_->get_primary_data(), params.parents,
                        is_root, is_exhausted, new_mst_edges);

    nb_mst_edges = gbbs_boruvka::add_new_mst_edges_to_mst_array(
        n, nb_mst_edges, new_mst_edges, params.edges, params.mst);
    gbbs_boruvka::update_parents(n, params.vertices_->get_primary_data(),
                                 params.parents);

    num_exhausted_vertices += identify_exhausted_vertices(
        n, is_exhausted, params.vertices_->get_primary_data(),
        exhausted_vertices, num_exhausted_vertices);
    n = compactify_vertices(n, params.vertices_, is_root);
    relabel_edges(m, edge_ids.get_primary_data(), params.edges, params.parents);
    m = filter_out_self_loops_(m, edge_ids);
    round++;
  }

  print("after loop");
  parallel_for(0, num_exhausted_vertices, [&](const auto& i) {
    params.vertices_->get_primary_data()[i + n] = exhausted_vertices[i];
  });
  n += num_exhausted_vertices;
  // for(std::size_t i = 0; i < n - num_exhausted_vertices; ++i) {
  //   std::cout << "active: " << params.vertices_->get_primary_data()[i] <<
  //   std::endl;
  // }
  // for(std::size_t i = n - num_exhausted_vertices; i < n; ++i) {
  //   std::cout << "exhausted: " << params.vertices_->get_primary_data()[i] <<
  //   std::endl;
  // }
  hybridMST::parallel_for(0, params.n_initial, [&](size_t i) {
    while (params.parents[i] != params.parents[params.parents[i]]) {
      params.parents[i] = params.parents[params.parents[i]];
    }
  });

  print("after parent update");
  return BoruvkaResult{nb_mst_edges, n};
}

template <typename Params> auto boruvka_filter_recursive(Params params) {
  mpi::MPIContext ctx;
  if (ctx.rank() == 0) {
    std::cout << params << std::endl;
  }
  if (params.vertex_range.n() <= 1) {
    return BoruvkaResult{0, params.vertex_range.n()};
    // return 0ul;
  }
  if (params.edges.size() <= 3 * params.n_initial) {
    const auto res = recursive_plain_filter_boruvka::boruvka(params);
    return res;
  }

  auto [light_edges, is_balanced, pivot] = localMST::partition(params.edges);
  if (!is_balanced) {
    const auto res = recursive_plain_filter_boruvka::boruvka(params);
    return res;
  }
  // PRINT_CONTAINER_WITH_INDEX(light_edges);
  // PRINT_CONTAINER_WITH_INDEX(heavy_edges);

  // **************************************************************
  // * recurse on light half
  // **************************************************************
  Params light_params = params;
  light_params.edges = Span(light_edges);
  auto num_light_edges = light_edges.size();
  auto num_heavy_edges = params.edges.size() - num_light_edges;
  const auto [num_mst_edges, num_active_vertices] =
      boruvka_filter_recursive(light_params);
  // **************************************************************
  // * filter
  // **************************************************************
  lotimer().start("filter");
  auto heavy_edges = localMST::filter(params, params.edges, pivot);
  lotimer().stop();
  if (ctx.rank() == 0) {
    std::cout << "light edges: " << num_light_edges
              << " heavy edges: " << num_heavy_edges
              << " filtered edges: " << heavy_edges.size()
              << " active vertices: " << num_active_vertices << std::endl;
  }
  Params heavy_params = params;
  heavy_params.mst += num_mst_edges;
  heavy_params.edges = heavy_edges;
  heavy_params.vertex_range.v_end =
      heavy_params.vertex_range.v_begin + num_active_vertices;
  // **************************************************************
  // * solve filtered half
  // **************************************************************
  lotimer().start("boruvka");
  auto res = recursive_plain_filter_boruvka::boruvka(heavy_params);
  lotimer().stop();
  res.num_mst_edges_found += num_mst_edges;
  return res;
}
} // namespace recursive_plain_filter_boruvka

template <typename EdgeType, typename Parent>
std::size_t boruvka_filter_plain(VertexRange_ vertex_range,
                                 Span<EdgeType> edges, Span<Parent> parents,
                                 GlobalEdgeId* mst) {

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
  PlainBoruvkaParameters<EdgeType, Parent, VId> params;
  params.n_initial = vertex_range.n();
  params.vertex_range = vertex_range;
  params.edges = edges;
  params.parents = parents;
  params.mst = mst;
  params.vertices_tmp_storage = vertices_tmp_storage.data();
  params.min_edges = min_edges;
  params.vertices_ = &vertices_;
  BoruvkaResult res =
      recursive_plain_filter_boruvka::boruvka_filter_recursive(params);
  if (ctx.rank() == 0) {
    lotimer().output(std::to_string(ctx.rank()));
  }
  lotimer().reset();
  return res.num_mst_edges_found;
}
} // namespace hybridMST
