#pragma once

#include "RQuick/RQuick.hpp"
#include "parlay/sequence.h"

#include "algorithms/algorithm_configurations.hpp"
#include "algorithms/base_case_mst_algos.hpp"
#include "algorithms/boruvka_allreduce.hpp"
#include "algorithms/hybrid_boruvka_substeps/duplicate_detection.hpp"
#include "algorithms/hybrid_boruvka_substeps/edge_consecutive.hpp"
#include "algorithms/hybrid_boruvka_substeps/edge_renaming.hpp"
#include "algorithms/hybrid_boruvka_substeps/get_ghost_representatives.hpp"
#include "algorithms/hybrid_boruvka_substeps/is_representative_computation.hpp"
#include "algorithms/hybrid_boruvka_substeps/local_kernelization.hpp"
#include "algorithms/hybrid_boruvka_substeps/minimum_edge_computation.hpp"
#include "algorithms/hybrid_boruvka_substeps/misc.hpp"
#include "algorithms/hybrid_boruvka_substeps/representative_computation.hpp"
#include "algorithms/input_edge_partitioning.hpp"
#include "algorithms/local_contraction/local_boruvka/local_mst_and_contraction.hpp"
#include "algorithms/local_contraction/local_contraction.hpp"
#include "datastructures/compression/difference_encoded_graph.hpp"
#include "datastructures/compression/uncompressed_graph.hpp"
#include "datastructures/distributed_graph.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

namespace hybridMST {

template <typename Graph, typename Container>

void print_with_idx(const Graph& graph, const Container& cont,
                    const std::string& desc) {
  std::cout << desc << std::endl;
  for (std::size_t i = 0; i < cont.size(); ++i) {
    std::cout << i << " global_id: " << graph.get_global_id(i) << " " << cont[i]
              << std::endl;
  }
}

struct TimePoint {
  TimePoint(std::string id_) : id{id_} {
    tp = std::chrono::steady_clock::now();
  }
  std::string id;
  std::chrono::steady_clock::time_point tp;
};

template <typename LocalRefEdges, typename Graph, typename GlobalRefEdges>
void visualise_local(const LocalRefEdges& local_ref_edges,
                     const GlobalRefEdges& global_ref_edges,
                     const non_init_vector<LocalEdgeId>& min_edges,
                     const non_init_vector<GlobalEdgeId> mst_ids,
                     const Graph& graph) {
  mpi::MPIContext ctx;
  ctx.execute_in_order([&]() {
    for (std::size_t i = 0; i < min_edges.size(); ++i) {
      const auto min_edge_idx = min_edges[i];
      std::cout << i << " " << local_ref_edges[min_edge_idx] << std::endl;
    }
  });

  const auto mst_edges = GetMstEdge::execute(global_ref_edges, mst_ids);
  SEQ_EX(ctx, PRINT_VECTOR(mst_edges););
}
// for debugging purposes only
template <typename Edges> void verify_edges(Edges& edges) {
  mpi::MPIContext ctx;
  Edge min_edge{edges.front().get_src(), edges.front().get_dst()};
  Edge max_edge{edges.back().get_src(), edges.back().get_dst()};

  VertexLocator_Split locator{min_edge, max_edge};
  SEQ_EX(ctx, std::cout << min_edge << " " << max_edge << std::endl;);

  auto filter = False_Predicate{};
  auto transformer = [&](const auto& edge, const std::size_t&) {
    const VId src = edge.get_src();
    const VId dst = edge.get_dst();
    const VId w = edge.get_weight();
    return WEdge{dst, src, w};
  };
  auto dst_computer = [&](const auto& edge, const std::size_t& i) {
    Edge e{edge.get_dst(), edge.get_src()};
    return locator.get_min_pe(e);
  };
  auto remote_edges =
      mpi::alltoall_combined(edges, filter, transformer, dst_computer).buffer;
  const auto is_equal = [](const auto& lhs, const auto& rhs) {
    return std::make_tuple(lhs.get_src(), lhs.get_dst(), lhs.get_weight()) ==
           std::make_tuple(rhs.get_src(), rhs.get_dst(), rhs.get_weight());
  };
  for (const auto edge : remote_edges) {
    auto it = std::lower_bound(
        edges.begin(), edges.end(), edge,
        [&](const auto& lhs, const WEdge& value) {
          return std::make_tuple(lhs.get_src(), lhs.get_dst(),
                                 lhs.get_weight()) <
                 std::make_tuple(value.get_src(), value.get_dst(),
                                 value.get_weight());
        });
    if (it == edges.end() || !is_equal(*it, edge)) {
      std::cout << ctx.rank() << " no remote for: " << edge << std::endl;
    }
  }
}

template <typename WEdgeIds>
inline void boruvka_core(WEdgeIds& augmented_edges,
                         non_init_vector<GlobalEdgeId>& mst_edge_ids) {
  hybridMST::mpi::MPIContext ctx;
  using WEdgeIdType = typename WEdgeIds::value_type;

  bool stop_loop = false;
  VId num_global_vertices = VID_UNDEFINED;
  for (std::size_t i = 0; /* true */; ++i) {
    const bool is_first_iteration = i == 0;
    get_timer().start("graph_init", i);
    REORDERING_BARRIER
    constexpr bool compactify = true;
    DistributedGraph<WEdgeIdType, compactify> graph(augmented_edges, i);
    if (is_first_iteration) {
      // cannot not query this value before, as it would create overhead
      // otherwise ...
      statistics_helper("graph_num_vertices_after_kernelization",
                        graph.local_n());
    }
    get_timer().start("_stop_criterion", i);
    REORDERING_BARRIER
    stop_loop =
        stop_boruvka(graph, 0, 0); // TODO change signature of stop_boruvka
    REORDERING_BARRIER
    get_timer().stop("_stop_criterion", i);
    if (stop_loop) {
      num_global_vertices = MakeVerticesConsecutive::execute(graph);
      // TODO check whether we want some statistics about the number of vertices
      // after making the ids consecutive
      (void)num_global_vertices;
      get_timer().stop("graph_init", i);
      break;
    }
    {
      REORDERING_BARRIER
      get_timer().stop("graph_init", i);

      statistics_helper("graph_num_vertices", graph.local_n(), i);
      statistics_helper("graph_num_edges", graph.edges().size(), i);
      get_timer().start("min_edges", i);
      REORDERING_BARRIER
      auto min_edge_ids_par = MinimumEdgeOpenMP::execute(graph);
      REORDERING_BARRIER
      get_timer().stop("min_edges", i);
      get_timer().start("is_representative", i);
      REORDERING_BARRIER
      auto is_rep = IsRepresentative_Push::compute_representatives_(
          min_edge_ids_par, graph);
      REORDERING_BARRIER
      get_timer().stop("is_representative", i);
      get_timer().start("add_mst_edges", i);
      REORDERING_BARRIER
      AddMstEdgesSeq::execute(graph, is_rep, min_edge_ids_par, mst_edge_ids);
      REORDERING_BARRIER
      get_timer().stop("is_representative", i);
      get_timer().stop("add_mst_edges", i);
      // get_timer().enable_measurements();
      get_timer().start("compute_representatives", i);
      REORDERING_BARRIER
      const auto rep_local_vertices =
          ComputeRepresentative::compute_representatives(graph, is_rep, i);
      REORDERING_BARRIER
      get_timer().stop("compute_representatives", i);
      // get_timer().disable_measurements();
      get_timer().start("get_ghost_representatives", i);
      REORDERING_BARRIER
      auto name_newName_ghost_vertices_ =
          ExchangeRepresentativesPush_Sort::execute(graph, rep_local_vertices,
                                                    i);
      REORDERING_BARRIER
      get_timer().stop("get_ghost_representatives", i);
      get_timer().start("rename_edges", i);

      // visualise_local(augmented_edges, edges, min_edge_ids_par, mst_edge_ids,
      //                 graph);
      REORDERING_BARRIER
      EdgeRenamer::rename_edges(graph, rep_local_vertices,
                                name_newName_ghost_vertices_);
      REORDERING_BARRIER
      get_timer().stop("rename_edges", i);

      get_timer().start("remove_self_loops", i);
      REORDERING_BARRIER
      EdgeProcessor::remove_self_loops(graph, i);
      REORDERING_BARRIER
      get_timer().stop("remove_self_loops", i);

      get_timer().start("redistribute_edges", i);
      REORDERING_BARRIER
      RedistributeViaSelectedSorter::redistribute(graph.edges());
      REORDERING_BARRIER
      get_timer().stop("redistribute_edges", i);

      get_timer().start("remove_duplicate_edges", i);
      REORDERING_BARRIER
      EdgeProcessor::remove_duplicates(graph, i);
      REORDERING_BARRIER
      get_timer().stop("remove_duplicate_edges", i);
    }
  }

  get_timer().start("base_case");
  REORDERING_BARRIER
  ips4o::parallel::sort(augmented_edges.begin(), augmented_edges.end(),
                        SrcDstWeightOrder<WEdgeIdType>{});
  std::vector<GlobalEdgeId> ids_gbbs;
  const std::size_t v_max =
      augmented_edges.empty() ? 0 : augmented_edges.back().get_src();
  const std::size_t n = mpi::allreduce_max(v_max) + 1;
  dense_boruvka_allreduce(n, augmented_edges, ids_gbbs);
  mst_edge_ids = combine(mst_edge_ids, ids_gbbs);
  REORDERING_BARRIER
  get_timer().stop("base_case");
}

template <typename AlgorithmConfigType, typename EdgeContainer>
inline WEdgeList boruvka_setup_and_preprocessing(
    EdgeContainer& edge_container,
    const AlgorithmConfigType& algorithm_configuration) {

  mpi::MPIContext ctx;
  get_timer().start("local_kernelization");
  non_init_vector<GlobalEdgeId> mst_edge_ids;
  auto augmented_edges = edge_container.get_WEdgeIds();
  // update_edge_ids(augmented_edges);
  // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(augmented_edges););
  // if (ctx.rank() == 0) {
  //   PRINT_VECTOR_WITH_INDEX(augmented_edges);
  // }
  const auto num_edge_before = mpi::allreduce_sum(augmented_edges.size());
  local_contraction_dispatcher(algorithm_configuration, augmented_edges,
                               mst_edge_ids);
  const auto num_edge_after = mpi::allreduce_sum(augmented_edges.size());
  if (ctx.rank() == 0) {
    std::cout << "num_edges_before: " << num_edge_before
              << " num_edge_after: " << num_edge_after << std::endl;
  }
  // DistributedGraph<typename AlgorithmConfigType::WEdgeIdType, true> graph(
  //     augmented_edges, 0);
  // SEQ_EX(ctx, std::cout << augmented_edges.front().get_src() << " "
  //                       << augmented_edges.back().get_src() << " "
  //                       << graph.local_n() << std::endl;);

  get_timer().stop("local_kernelization");

  boruvka_core(augmented_edges, mst_edge_ids);

  get_timer().start("send_mst_edges_back");
  auto mst_edges =
      GetMstEdge::execute(edge_container.get_WEdges(), mst_edge_ids);
  get_timer().stop("send_mst_edges_back");
  return mst_edges;
}

template <typename InputEdges, typename AlgorithmConfigType>
inline WEdgeList boruvka(InputEdges input_edges,
                         const AlgorithmConfigType& algorithm_configuration) {
  using WEdgeType = typename AlgorithmConfigType::WEdgeType;
  using WEdgeIdType = typename AlgorithmConfigType::WEdgeIdType;
  mpi::MPIContext ctx;

  get_timer().add("graph_num_edges_initial", 0, input_edges.size(),
                  Timer::DatapointsOperation::ID);
  const std::size_t num_local_vertices = get_number_local_vertices(input_edges);
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
    return boruvka_setup_and_preprocessing(edge_container,
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
    return boruvka_setup_and_preprocessing(edge_container,
                                           algorithm_configuration);
  }
  default:
    return WEdgeList{};
  }
}

} // namespace hybridMST
