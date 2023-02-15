#pragma once

#include <cstdint>
#include <mpi/gather.hpp>

#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "duplicate_detection.hpp"
#include "edge_renaming.hpp"
#include "get_ghost_representatives.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/scan.hpp"

namespace hybridMST {

struct MakeVerticesConsecutive {
  template <typename Graph>
  static std::size_t get_num_distinct_vertices(const Graph& graph) {
    auto& locator = graph.locator();
    mpi::MPIContext ctx;
    // SEQ_EX(ctx, PRINT_VECTOR(graph.local_n());
    // PRINT_VECTOR(locator.is_home_of_v_min););
    if (graph.local_n() == 0)
      return 0ull;
    if (locator.is_home_of_v_min())
      return graph.local_n();
    return graph.local_n() - 1;
  }

  template <typename Graph> static std::size_t execute(Graph& graph) {
    mpi::MPIContext ctx;
    auto& locator = graph.locator();

    const std::size_t num_distinct_vertices = get_num_distinct_vertices(graph);
    const std::size_t num_vertices = graph.local_n();
    const std::size_t global_prefix =
        mpi::exscan_sum(num_distinct_vertices, ctx, std::size_t{0});
    const std::size_t global_prefix_adjusted =
        locator.is_home_of_v_min() ? global_prefix : global_prefix - 1;
    const std::size_t new_global_num_vertices =
        mpi::allreduce_sum(num_distinct_vertices);
    non_init_vector<VId> local_id_new_name(num_vertices);
    parallel_for(0, num_vertices, [&](std::size_t i) {
      local_id_new_name[i] = global_prefix_adjusted + i;
    });

    // SEQ_EX(ctx, PRINT_VECTOR(graph.edges());
    // PRINT_VECTOR(local_id_new_name););
    const int round = 999'999;
    auto name_newName_ghost_vertices_ =
        ExchangeRepresentativesPush_Sort::execute(graph, local_id_new_name,
                                                  round);
    EdgeRenamer::rename_edges(graph, local_id_new_name,
                              name_newName_ghost_vertices_);

    EdgeProcessor::remove_self_loops(graph.edges(), round);
    // SEQ_EX(ctx, PRINT_VECTOR(graph.edges()););
    return new_global_num_vertices;
  }
};
struct MakeVerticesConsecutive_SmallVertexSize
    : public MakeVerticesConsecutive {
  template <typename Graph> static non_init_vector<VId> execute_(Graph& graph) {
    mpi::MPIContext ctx;
    auto& locator = graph.locator();

    const std::size_t num_distinct_vertices = get_num_distinct_vertices(graph);
    const std::size_t num_vertices = graph.local_n();
    const std::size_t global_prefix =
        mpi::exscan_sum(num_distinct_vertices, ctx, std::size_t{0});
    const std::size_t global_prefix_adjusted =
        locator.is_home_of_v_min() ? global_prefix : global_prefix - 1;
    const std::size_t new_global_num_vertices =
        mpi::allreduce_sum(num_distinct_vertices);
    non_init_vector<VId> local_id_new_name(num_vertices);
    parallel_for(0, num_vertices, [&](std::size_t i) {
      local_id_new_name[i] = global_prefix_adjusted + i;
    });
    non_init_vector<VId> compactifiedId_prevId(num_distinct_vertices);
    const bool is_not_home_of_v_min = !locator.is_home_of_v_min();
    parallel_for(0, num_distinct_vertices, [&](std::size_t i) {
      compactifiedId_prevId[i] = graph.get_global_id(i + is_not_home_of_v_min);
    });
    compactifiedId_prevId = mpi::allgatherv(compactifiedId_prevId);

    // SEQ_EX(ctx, PRINT_VECTOR(local_id_new_name););
    const int round = 999'999;
    auto name_newName_ghost_vertices_ =
        ExchangeRepresentativesPush_Sort::execute(graph, local_id_new_name,
                                                  round);

    growt::GlobalVIdMap<VId> grow_map(compactifiedId_prevId.size() * 1.25);
    parallel_for(0, compactifiedId_prevId.size(), [&](std::size_t i) {
      const auto& prevId = compactifiedId_prevId[i];
      grow_map.insert(prevId + 1, i);
    });
    EdgeRenamer::rename_edges(graph, local_id_new_name,
                              name_newName_ghost_vertices_);
    EdgeProcessor::remove_self_loops(graph.edges(), round);
    return compactifiedId_prevId;
  }
};

struct MakeVerticesConsecutive_WithoutAllgatherv
    : public MakeVerticesConsecutive {
  template <typename Graph> static non_init_vector<VId> execute(Graph& graph) {
    mpi::MPIContext ctx;
    auto& locator = graph.locator();

    const std::size_t num_distinct_vertices = get_num_distinct_vertices(graph);
    const std::size_t num_vertices = graph.local_n();
    const std::size_t global_prefix =
        mpi::exscan_sum(num_distinct_vertices, ctx, std::size_t{0});
    const std::size_t global_prefix_adjusted =
        locator.is_home_of_v_min() ? global_prefix : global_prefix - 1;
    non_init_vector<VId> local_id_new_name(num_vertices);
    parallel_for(0, num_vertices, [&](std::size_t i) {
      local_id_new_name[i] = global_prefix_adjusted + i;
    });
    non_init_vector<VId> compactifiedId_prevId(num_distinct_vertices);
    const bool is_not_home_of_v_min = !locator.is_home_of_v_min();
    parallel_for(0, num_distinct_vertices, [&](std::size_t i) {
      compactifiedId_prevId[i] = graph.get_global_id(i + is_not_home_of_v_min);
    });
    const PEID root = 0;
    compactifiedId_prevId =
        mpi::two_level::gatherv(compactifiedId_prevId, root);

    const int round = 999'999;
    auto name_newName_ghost_vertices_ =
        ExchangeRepresentativesPush_Sort::execute(graph, local_id_new_name,
                                                  round);

    growt::GlobalVIdMap<VId> grow_map(compactifiedId_prevId.size() * 1.25);
    parallel_for(0, compactifiedId_prevId.size(), [&](std::size_t i) {
      const auto& prevId = compactifiedId_prevId[i];
      grow_map.insert(prevId + 1, i);
    });
    EdgeRenamer::rename_edges(graph, local_id_new_name,
                              name_newName_ghost_vertices_);
    EdgeProcessor::remove_self_loops(graph.edges(), round);
    return compactifiedId_prevId;
  }
};
} // namespace hybridMST
