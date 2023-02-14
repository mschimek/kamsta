#pragma once

#include "RQuick/RQuick.hpp"
#include "parlay/sequence.h"

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
#include "algorithms/local_contraction/local_contraction.hpp"
#include "datastructures/concurrent_lookup_map.hpp";
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
template <typename Edges> void verify_edges(Edges& edges) {
  using EdgeType = typename Edges::value_type;
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

template <typename CompressedGraph>
inline WEdgeList boruvka(const CompressedGraph& compressed_graph,
                         VertexRange range,
                         std::size_t local_kernelization_level) {
  // get_timer().disable_measurements();
  using WEdgeIdType = typename CompressedGraph::WEdgeIdType;
  hybridMST::mpi::MPIContext ctx;

  // SEQ_EX(ctx, PRINT_VECTOR(edges););
  // SEQ_EX(ctx, PRINT_VAR(edges.size()););

  const std::size_t initial_n_real = mpi::allreduce_max(range.second) + 1;
  const std::size_t num_worker = ctx.size() * ctx.threads_per_mpi_process();
  const std::size_t initial_m =
      mpi::allreduce_sum(compressed_graph.num_local_edges()) / num_worker;
  const std::size_t initial_n = initial_n_real / num_worker;

  get_timer().add("graph_num_edges_initial", 0,
                  compressed_graph.num_local_edges(),
                  Timer::DatapointsOperation::ID);
  get_timer().add("graph_num_vertices_initial", 0,
                  (range.second - range.first + 1),
                  Timer::DatapointsOperation::ID);
  non_init_vector<GlobalEdgeId> mst_edge_ids;

  get_timer().start("local_kernelization");
  // auto augmented_edges = LocalKernelization::execute(range, Span<const
  // WEdge>(edges.data(), edges.size()));
  auto augmented_edges =
      compressed_graph.get_WEdgeIds(); // LocalKernelization::execute<const
                                       // WEdge, WEdgeId>(range, Span(edges));
  // verify_edges(augmented_edges);
  if (local_kernelization_level == 1) {
    local_contraction(augmented_edges, mst_edge_ids);
  }
  get_timer().stop("local_kernelization");
  get_timer().add("graph_num_edges_after_kernelization", 0,
                  augmented_edges.size(), Timer::DatapointsOperation::ID);

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
      get_timer().add("graph_num_vertices_after_kernelization", 0,
                      graph.local_n(), Timer::DatapointsOperation::ID);
    }
    get_timer().start("_stop_criterion", i);
    REORDERING_BARRIER
    stop_loop = stop_boruvka(graph, initial_m, initial_n);
    REORDERING_BARRIER
    get_timer().stop("_stop_criterion", i);
    if (stop_loop) {
      num_global_vertices = MakeVerticesConsecutive::execute(graph);
      get_timer().stop("graph_init", i);
      break;
    }
    {
      REORDERING_BARRIER
      get_timer().stop("graph_init", i);

      get_timer().add("graph_num_edges", i, graph.edges().size(),
                      Timer::DatapointsOperation::ID);
      get_timer().add("graph_num_vertices", i, graph.local_n(),
                      Timer::DatapointsOperation::ID);
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
  auto mst_edge_ids_combined = combine(mst_edge_ids, ids_gbbs);
  REORDERING_BARRIER
  get_timer().stop("base_case");
  get_timer().start("send_mst_edges_back");
  REORDERING_BARRIER
  auto mst_edges =
      GetMstEdge::execute(compressed_graph.get_WEdges(), mst_edge_ids_combined);
  REORDERING_BARRIER
  get_timer().stop("send_mst_edges_back");
  // SEQ_EX(ctx, std::sort(mst_edges.begin(), mst_edges.end(),
  //                       SrcDstWeightOrder<WEdge>{});
  //        PRINT_VECTOR(mst_edges););

  return mst_edges;
}
} // namespace hybridMST
