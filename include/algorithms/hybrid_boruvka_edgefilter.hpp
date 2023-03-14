#pragma once
#include "RQuick/RQuick.hpp"
#include "algorithms/algorithm_configurations.hpp"
#include "algorithms/base_case_mst_algos.hpp"
#include "algorithms/boruvka_allreduce.hpp"
#include "algorithms/filter_boruvka_substeps/filter_helpers.hpp"
#include "algorithms/filter_boruvka_substeps/get_new_labels.hpp"
#include "algorithms/hybrid_boruvka_substeps/edge_consecutive.hpp"
#include "algorithms/hybrid_boruvka_substeps/edge_renaming.hpp"
#include "algorithms/hybrid_boruvka_substeps/get_ghost_representatives.hpp"
#include "algorithms/hybrid_boruvka_substeps/is_representative_computation.hpp"
#include "algorithms/hybrid_boruvka_substeps/minimum_edge_computation.hpp"
#include "algorithms/hybrid_boruvka_substeps/misc.hpp"
#include "algorithms/hybrid_boruvka_substeps/representative_computation.hpp"
#include "algorithms/local_contraction/local_contraction.hpp"
#include "algorithms/twolevel_sorting/twolevel_sorting.hpp"
#include "datastructures/compression/difference_encoded_graph.hpp"
#include "datastructures/compression/uncompressed_graph.hpp"
#include "datastructures/distributed_graph.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "parlay/sequence.h"
#include "util/macros.hpp"
#include "util/timer.hpp"
#include <util/memory_utils.hpp>

namespace hybridMST {

inline std::size_t id = 0;
struct LeftoverManager {
  void recurse_left() { ++counter; }
  void come_back() { --counter; }
  bool can_handle_leftovers() { return counter > 0; }
  int counter = 0;
};

inline bool do_perform_right_recursion(std::size_t num_edges) {
  mpi::MPIContext ctx;
  const auto global_num_edges = mpi::allreduce_sum(num_edges);
  const std::size_t threshold = std::max(ctx.size(), 10'000);
  return (global_num_edges / ctx.size()) >= threshold;
}

inline std::size_t requested_new_label = 0;
template <typename EdgeType, bool compactify_graph>
inline bool filter_boruvka_step(int round, non_init_vector<EdgeType>& edges,
                                non_init_vector<GlobalEdgeId>& mst_edge_ids,
                                ParentArray& parent_array, const VId initial_n,
                                const VId initial_m) {
  mpi::MPIContext ctx;
  get_timer().start("graph_init", round);
  REORDERING_BARRIER
  DistributedGraph<EdgeType, compactify_graph> graph(edges, round);
  REORDERING_BARRIER
  get_timer().stop("graph_init", round);
  get_timer().start("_stop_criterion", round);
  const bool stop_boruvka_ = stop_boruvka(graph, initial_m, initial_n);
  REORDERING_BARRIER
  get_timer().stop("_stop_criterion", round);
  if (stop_boruvka_)
    return true;
  get_timer().start("min_edges", round);
  REORDERING_BARRIER
  auto min_edge_ids_par = MinimumEdgeOpenMP::execute(graph);
  REORDERING_BARRIER
  get_timer().stop("min_edges", round);
  get_timer().start("is_representative", round);
  REORDERING_BARRIER
  auto is_rep =
      IsRepresentative_Push::compute_representatives_(min_edge_ids_par, graph);
  // auto is_rep_ =
  //     IsRepresentative_Push::compute_representatives(min_edge_ids_par,
  //     graph);
  REORDERING_BARRIER
  get_timer().stop("is_representative", round);
  get_timer().start("add_mst_edges", round);
  REORDERING_BARRIER
  AddMstEdgesSeq::execute(graph, is_rep, min_edge_ids_par, mst_edge_ids);
  REORDERING_BARRIER
  get_timer().stop("add_mst_edges", round);
  // get_timer().enable_measurements();
  get_timer().start("compute_representatives", round);
  REORDERING_BARRIER
  const auto rep_local_vertices =
      ComputeRepresentative::compute_representatives(graph, is_rep, round);
  REORDERING_BARRIER
  get_timer().stop("compute_representatives", round);

  get_timer().start("update_parent_array", round);
  REORDERING_BARRIER
  UpdateParentArray::execute(graph, parent_array, rep_local_vertices);
  REORDERING_BARRIER
  get_timer().stop("update_parent_array", round);

  get_timer().start("get_ghost_representatives", round);
  REORDERING_BARRIER
  auto name_newName_ghost_vertices_ = ExchangeRepresentativesPush_Sort::execute(
      graph, rep_local_vertices, round);
  REORDERING_BARRIER
  get_timer().stop("get_ghost_representatives", round);
  get_timer().start("rename_edges", round);
  REORDERING_BARRIER
  EdgeRenamer::rename_edges(graph, rep_local_vertices,
                            name_newName_ghost_vertices_);
  REORDERING_BARRIER
  get_timer().stop("rename_edges", round);

  get_timer().start("remove_self_loops", round);
  REORDERING_BARRIER
  EdgeProcessor::remove_self_loops(graph, round);
  REORDERING_BARRIER
  get_timer().stop("remove_self_loops", round);

  get_timer().start("redistribute_edges", round);
  REORDERING_BARRIER
  RedistributeViaSelectedSorter::redistribute(graph.edges());
  REORDERING_BARRIER
  get_timer().stop("redistribute_edges", round);

  get_timer().start("remove_duplicate_edges", round);
  REORDERING_BARRIER
  EdgeProcessor::remove_duplicates(graph, round);
  REORDERING_BARRIER
  get_timer().stop("remove_duplicate_edges", round);
  return false;
}

template <typename EdgeType>
inline void filter_boruvka_basecase(non_init_vector<EdgeType>& edges,
                                    non_init_vector<GlobalEdgeId>& mst_edge_ids,
                                    ParentArray& parent_array, VId global_n,
                                    int round) {

  hybridMST::mpi::MPIContext ctx;
  const std::size_t num_worker = ctx.size() * ctx.threads_per_mpi_process();
  const std::size_t initial_m = mpi::allreduce_sum(edges.size()) / num_worker;
  const std::size_t initial_n = global_n / num_worker;

  non_init_vector<VId> compactifiedId_prevId;
  for (;; ++round) {
    memory_stats().print("boruvka - round: " + std::to_string(round));
    constexpr bool compactify_graph = true;
    const bool stop_boruvka = filter_boruvka_step<EdgeType, compactify_graph>(
        round, edges, mst_edge_ids, parent_array, initial_m, initial_n);
    if (stop_boruvka) {
      ++round;
      get_timer().start("make_graph_consecutive", round);
      // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(edges););
      REORDERING_BARRIER
      DistributedGraph<EdgeType, compactify_graph> graph(edges, round);
      compactifiedId_prevId =
          MakeVerticesConsecutive_WithoutAllgatherv::execute(graph);
      REORDERING_BARRIER
      get_timer().stop("make_graph_consecutive", round);
      break;
    }
  }

  get_timer().start("base_case");
  REORDERING_BARRIER
  const std::size_t num_global_vertices =
      mpi::bcast(compactifiedId_prevId.size(), 0); // data is only on root
  std::vector<GlobalEdgeId> ids_gbbs;

  dense_boruvka_allreduce(num_global_vertices, edges, ids_gbbs,
                          compactifiedId_prevId, parent_array);
  append_second_to_first(mst_edge_ids, ids_gbbs);
  get_timer().stop("base_case");
}

template <typename EdgeType>
inline void boruvka_recursive_setup(int round, non_init_vector<EdgeType>& edges,
                                    ParentArray& parent_array,
                                    Weight threshold_weight, LabelCache& cache,
                                    non_init_vector<EdgeType>& leftover_edges) {
  mpi::MPIContext ctx;
  // SEQ_EX(ctx, PRINT_VAR(parent_array); PRINT_VECTOR(edges););

  get_timer().start("cache_write_init", round);
  REORDERING_BARRIER
  cache.rename_edges(edges, threshold_weight);
  REORDERING_BARRIER
  // SEQ_EX(ctx, PRINT_VECTOR(edges););
  get_timer().stop("cache_write_init", round);
  get_timer().start_phase("recursive_relabel");
  get_timer().start("retrieve_rename_edges_init", round);
  REORDERING_BARRIER
  // rename_edges(round, Span(edges.data(), edges.size()), parent_array,
  //              threshold_weight, cache);
  RelabelViaPush::execute(round, edges, parent_array, threshold_weight);
  REORDERING_BARRIER
  get_timer().stop("retrieve_rename_edges_init", round);
  get_timer().stop_phase();

  get_timer().start("remove_self_loops_init", round);
  REORDERING_BARRIER
  EdgeProcessor::remove_self_loops(edges, round);
  REORDERING_BARRIER
  get_timer().stop("remove_self_loops_init", round);

  if (!leftover_edges.empty()) {
    non_init_vector<EdgeType> tmp(edges.size() + leftover_edges.size());
    parallel_for(0, edges.size(),
                 [&](const std::size_t i) { tmp[i] = edges[i]; });
    parallel_for(0, leftover_edges.size(), [&](const std::size_t i) {
      tmp[i + edges.size()] = leftover_edges[i];
    });
    leftover_edges.clear();
    edges = std::move(tmp);
  }

  get_timer().start("redistribute_edges_init1", round);
  REORDERING_BARRIER
  RedistributeViaSelectedSorter::redistribute(edges);
  REORDERING_BARRIER
  get_timer().stop("redistribute_edges_init1", round);

  get_timer().start("remove_duplicate_edges_init", round);
  REORDERING_BARRIER
  EdgeProcessor::remove_duplicates(edges, round);
  REORDERING_BARRIER
  get_timer().stop("remove_duplicate_edges_init", round);
}

template <typename EdgeType>
inline void boruvka_recursive(
    int& round, int depth, std::size_t global_n, std::size_t global_m,
    non_init_vector<EdgeType>& edges, ParentArray& parent_array,
    non_init_vector<GlobalEdgeId>& mst_edge_ids, Weight threshold_weight,
    LabelCache& cache, std::size_t filter_threshold,
    non_init_vector<EdgeType>& leftover_edges,
    LeftoverManager& leftover_manager, std::uint64_t initial_global_n,
    bool stop_recursion = false) {
  mpi::MPIContext ctx;
  const auto id_copy = id;
  ++id;

  const bool no_more_edges = global_m == 0;
  const bool no_more_vertices = global_n <= 1;
  if (no_more_edges || no_more_vertices) {
    return;
  }
  const double avg_degree = global_m / static_cast<double>(initial_global_n);
  const std::size_t threshold_size_edges = 1000;
  const bool is_graph_small = global_m <= (ctx.size() * threshold_size_edges);
  if (avg_degree <= filter_threshold || stop_recursion || is_graph_small) {
    // base case
    std::size_t num_mst_edges_found = mst_edge_ids.size();
    get_timer().start("filter_basecase", round);
    filter_boruvka_basecase(edges, mst_edge_ids, parent_array, global_n, round);
    get_timer().stop("filter_basecase", round);
    get_timer().start("parent_array_shortcut", round);
    parent_array.shortcut();
    get_timer().stop("parent_array_shortcut", round);
    num_mst_edges_found = mst_edge_ids.size() - num_mst_edges_found;
    const auto sum_mst_edges_found = mpi::allreduce_sum(num_mst_edges_found);
    if (ctx.rank() == 0) {
      PRINT_VAR(round);
      PRINT_VAR(sum_mst_edges_found);
    }
    return;
  }
  get_timer().start("partition_edges", round);
  REORDERING_BARRIER
  Weight pivot = WEIGHT_INF;
  stop_recursion = false;
  auto light_half = partition_edges(edges, pivot, stop_recursion);
  const auto num_heavy_edges = edges.size() - light_half.size();
  if (ctx.rank() == 0) {
    std::cout << "pivot: " << pivot << " threshold: " << threshold_weight
              << " depth: " << depth << std::endl;
  }
  REORDERING_BARRIER
  get_timer().stop("partition_edges", round);
  round += 20;
  const std::size_t global_m_light_edges =
      mpi::allreduce_sum(light_half.size());
  leftover_manager.recurse_left();
  boruvka_recursive(round, depth + 1, global_n, global_m_light_edges,
                    light_half, parent_array, mst_edge_ids, threshold_weight,
                    cache, filter_threshold, leftover_edges, leftover_manager,
                    initial_global_n, stop_recursion);
  leftover_manager.come_back();
  get_timer().start("boruvka_recursive_setup", round);
  dump(light_half);
  REORDERING_BARRIER
  const std::size_t global_n_filtered = parent_array.global_num_vertices();

  get_timer().add("num_vertices_before_light_mst", id_copy, global_n,
                  {Timer::DatapointsOperation::MAX});
  get_timer().add("num_vertices_after_light_mst", id_copy, global_n_filtered,
                  {Timer::DatapointsOperation::MAX});
  if (global_n_filtered <= 1)
    return;

  if (ctx.rank() == 0) {
    std::cout << "pivot: " << pivot << " threshold: " << threshold_weight
              << " depth: " << depth << std::endl;
  }
  boruvka_recursive_setup(round, edges, parent_array, pivot, cache,
                          leftover_edges);

  REORDERING_BARRIER
  get_timer().stop("boruvka_recursive_setup", round);
  const std::size_t global_m_filtered = mpi::allreduce_sum(edges.size(), ctx);
  if (ctx.rank() == 0) {
    PRINT_VAR(round);
    PRINT_VAR(depth);
    requested_new_label += global_m - global_m_light_edges;
    PRINT_VAR(global_m - global_m_filtered);
    PRINT_VAR(global_n_filtered);
  }
  get_timer().add("num_heavy_edges", id_copy, num_heavy_edges,
                  {Timer::DatapointsOperation::ID});
  get_timer().add("num_heavy_edges_filtered", id_copy, edges.size(),
                  {Timer::DatapointsOperation::ID});
  round += 20;
  const bool perform_recursion = (!leftover_manager.can_handle_leftovers()) ||
                                 do_perform_right_recursion(edges.size());
  if (perform_recursion) {
    boruvka_recursive(round, depth + 1, global_n_filtered, global_m_filtered,
                      edges, parent_array, mst_edge_ids, pivot, cache,
                      filter_threshold, leftover_edges, leftover_manager,
                      initial_global_n);
  } else {
    leftover_edges = std::move(edges);
  }
}

template <typename Edges>
inline void init_parent_array(Edges& edges, ParentArray& parent_array,
                              VId num_local_vertices) {
  mpi::MPIContext ctx;
  // SEQ_EX(ctx, PRINT_VAR(num_local_vertices););
  parlay::hashtable<parlay::hash_numeric<VId>> table(
      num_local_vertices * 1.1, parlay::hash_numeric<VId>{});
  parallel_for(0, edges.size(), [&](std::size_t i) {
    const auto edge = edges[i];
    table.insert(edge.get_src());
  });
  auto entries = table.entries();
  // SEQ_EX(ctx, PRINT_VAR(entries.size()););
  non_init_vector<ParentArray::VertexHasEdges> vertices_with_edges(
      entries.size());
  parallel_for(0, entries.size(), [&](std::size_t i) {
    vertices_with_edges[i] = ParentArray::VertexHasEdges(entries[i], 1);
  });
  // SEQ_EX(ctx, PRINT_VAR(vertices_with_edges.size()););
  parent_array.set_non_isolated_vertices(vertices_with_edges);
}

template <typename WEdgeIds>
inline void filter_boruvka_core(WEdgeIds& augmented_edges,
                                ParentArray& parent_array,
                                LabelCache& label_cache,
                                non_init_vector<GlobalEdgeId>& mst_edge_ids,
                                std::size_t global_n, std::size_t global_m,
                                std::size_t filter_threshold) {
  using WEdgeIdType = typename WEdgeIds::value_type;
  mpi::MPIContext ctx;
  int round = 0;
  int depth = 0;
  id = 0;
  non_init_vector<WEdgeIdType> leftover_edges;
  LeftoverManager leftover_manager;
  const VId initial_global_n = global_n;
  boruvka_recursive(round, depth, global_n, global_m, augmented_edges,
                    parent_array, mst_edge_ids, 0, label_cache,
                    filter_threshold, leftover_edges, leftover_manager, initial_global_n);
  dump(augmented_edges);
}

template <typename AlgorithmConfigType, typename EdgeContainer>
inline WEdgeList filter_boruvka_setup_and_preprocessing(
    EdgeContainer& edge_container,
    const AlgorithmConfigType& algorithm_configuration) {
  mpi::MPIContext ctx;

  get_timer().start("local_kernelization");
  auto augmented_edges = edge_container.get_WEdgeIds();

  print_statistics(augmented_edges);
  const std::size_t num_local_vertices =
      get_number_local_vertices(augmented_edges);
  const std::size_t global_n =
      mpi::allreduce_max(get_max_local_vertex(augmented_edges)) + 1;
  const std::size_t global_m = mpi::allreduce_sum(augmented_edges.size());
  non_init_vector<GlobalEdgeId> mst_edge_ids;
  mst_edge_ids.reserve(10 + num_local_vertices * 2);
  LabelCache cache(0);
  get_timer().start("parent_array_init");
  ParentArray parent_array(global_n);
  init_parent_array(augmented_edges, parent_array, num_local_vertices);
  get_timer().stop("parent_array_init");

  local_contraction_dispatcher(algorithm_configuration, augmented_edges,
                               mst_edge_ids, parent_array);
  get_timer().stop("local_kernelization");

  filter_boruvka_core(augmented_edges, parent_array, cache, mst_edge_ids,
                      global_n, global_m,
                      algorithm_configuration.filter_threshold);

  get_timer().start("send_mst_edges_back");
  auto mst_edges =
      GetMstEdge::execute(edge_container.get_WEdges(), mst_edge_ids);
  const std::size_t global_n_filtered = parent_array.global_num_vertices();
  if (ctx.rank() == 0) {
    PRINT_VAR(requested_new_label);
    PRINT_VAR(global_m);
    PRINT_VAR(global_n_filtered);
  }
  requested_new_label = 0;
  get_timer().stop("send_mst_edges_back");
  return mst_edges;
}

template <typename InputEdges, typename AlgorithmConfigType>
inline WEdgeList
filter_boruvka(InputEdges input_edges,
               const AlgorithmConfigType& algorithm_configuration) {
  using WEdgeType = typename AlgorithmConfigType::WEdgeType;
  using WEdgeIdType = typename AlgorithmConfigType::WEdgeIdType;
  mpi::MPIContext ctx;
  get_timer().add("graph_num_edges_initial", 0, input_edges.size(),
                  Timer::DatapointsOperation::ID);
  const std::size_t num_local_vertices =
      input_edges.empty()
          ? 0
          : (input_edges.back().get_src() - input_edges.front().get_src()) + 1;
  get_timer().add("graph_num_vertices_initial", 0, (num_local_vertices),
                  Timer::DatapointsOperation::ID);

  const std::size_t edge_offset =
      hybridMST::mpi::exscan_sum(input_edges.size(), ctx, 0ul);

  switch (algorithm_configuration.compression) {
  case Compression::NO_COMPRESSION: {
    get_timer().start("edge_setup");
    UncompressedGraph<WEdgeType, WEdgeIdType> edge_container(
        std::move(input_edges), edge_offset);
    get_timer().stop("edge_setup");
    return filter_boruvka_setup_and_preprocessing(edge_container,
                                                  algorithm_configuration);
  }
  case Compression::SEVEN_BIT_DIFF_ENCODING: {
    get_timer().start("edge_setup");
    VertexRange range(input_edges.front().get_src(),
                      input_edges.back().get_src());
    hybridMST::DifferenceEncodedGraph<WEdgeType, WEdgeIdType> edge_container(
        input_edges, ctx.threads_per_mpi_process(), edge_offset, range);
    dump(input_edges);
    get_timer().stop("edge_setup");
    return filter_boruvka_setup_and_preprocessing(edge_container,
                                                  algorithm_configuration);
  }
  default:
    return WEdgeList{};
  }
}
} // namespace hybridMST
