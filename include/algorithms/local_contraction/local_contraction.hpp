#pragma once

#include <algorithm>

#include "parlay/hash_table.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/gbbs_reimplementation.hpp"
#include "algorithms/hybrid_boruvka_substeps/duplicate_detection.hpp"
#include "algorithms/hybrid_boruvka_substeps/edge_renaming.hpp"
#include "algorithms/hybrid_boruvka_substeps/get_ghost_representatives.hpp"
#include "algorithms/hybrid_boruvka_substeps/misc.hpp"
#include "algorithms/local_contraction/mst_on_local_edgs.hpp"
#include "algorithms/local_contraction/utils.hpp"
#include "datastructures/distributed_graph.hpp"
#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/context.hpp"
#include "mpi/send_recv.hpp"
#include "shared_mem_parallel.hpp"
#include "util/macros.hpp"
#include "util/utils.hpp"

namespace hybridMST {

namespace internal_local_contraction {

template <typename Edges, typename FwdIter>
void process_first_interval_on(bool process_lower_half, Edges& edges,
                               FwdIter end_first_interval,
                               FwdIter begin_last_interval) {

  mpi::MPIContext ctx;
  using EdgeType = typename Edges::value_type;
  Edges send_edges;
  int partner = -1;
  if (process_lower_half) {
    std::size_t send_size = std::distance(edges.begin(), end_first_interval);
    send_edges.resize(send_size);
    std::copy(edges.begin(), end_first_interval, send_edges.begin());
    partner = std::max(0, ctx.rank() - 1);
  } else {
    std::size_t send_size = std::distance(begin_last_interval, edges.end());
    send_edges.resize(send_size);
    std::copy(begin_last_interval, edges.end(), send_edges.begin());
    partner = std::min(ctx.size() - 1, ctx.rank() + 1);
  }
  Edges recv_edges = mpi::send_recv_v(partner, partner, send_edges);
  if (partner == ctx.rank())
    return; // first or last PE on v_begin or v_last respectively

  Edges merged_edges(send_edges.size() + recv_edges.size());
  std::merge(send_edges.begin(), send_edges.end(), recv_edges.begin(),
             recv_edges.end(), merged_edges.begin(),
             SrcDstWeightOrder<EdgeType>{});
  auto begin_merged_edges =
      process_lower_half ? std::next(merged_edges.begin(), recv_edges.size())
                         : merged_edges.begin();
  auto end_merged_edges =
      process_lower_half ? merged_edges.end()
                         : std::next(merged_edges.begin(), send_edges.size());
  auto output = process_lower_half ? edges.begin() : (begin_last_interval);
  std::copy(begin_merged_edges, end_merged_edges, output);
}

template <typename Edges> void reorder_splitted_edges(Edges& edges) {
  uint8_t is_sorting_necessary =
      edges.empty() || (edges.front().get_src() == edges.back().get_src());
  is_sorting_necessary = mpi::allreduce_max(is_sorting_necessary);
  if (is_sorting_necessary) {
    // if there are no edges or there is the possibility that one vertex could
    // be split over more than two PEs perform a complete sorting to restore the
    // invariants of the following MST steps, i.e. globally sorted edges array
    // with respect to (src,dst,weight)
    RedistributeViaPartitioning::redistribute(edges);
    return;
  }
  const VId v_begin = edges.front().get_src();
  auto it_v_past_v_begin =
      std::lower_bound(edges.begin(), edges.end(), (v_begin + 1),
                       [](const auto& edge, const VId next_vertex) {
                         return edge.get_src() < next_vertex;
                       });
  MPI_ASSERT_(it_v_past_v_begin != edges.end(),
              ""); // should not happen as of the is_sorting_necessary check
  const VId v_last = edges.back().get_src();
  const VId v_before_last = v_last - 1;
  auto it_v_before_last =
      std::lower_bound(edges.rbegin(), edges.rend(), v_before_last,
                       [](const auto& edge, const VId prev_vertex) {
                         return edge.get_src() > prev_vertex;
                       });
  MPI_ASSERT_(it_v_before_last != edges.rend(),
              ""); // should not happen as of the is_sorting_necessary check
  mpi::MPIContext ctx;
  // SEQ_EX(ctx, PRINT_VAR(*(it_v_past_v_begin));
  // PRINT_VAR(*(it_v_before_last));
  //        PRINT_CONTAINER_WITH_INDEX(edges););

  const bool rank_is_even = ctx.rank() % 2 == 0;
  auto end_first_interval = it_v_past_v_begin;
  auto begin_last_interval =
      std::prev(edges.end(), std::distance(edges.rbegin(), it_v_before_last));
  process_first_interval_on(rank_is_even, edges, end_first_interval,
                            begin_last_interval);
  process_first_interval_on(!rank_is_even, edges, end_first_interval,
                            begin_last_interval);
  // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(edges););
}

template <typename Edges, typename MstEdgeIds>
auto execute_contraction(Edges& edges, const VertexRange_ vertex_range,
                         MstEdgeIds& mst_edge_ids) {
  using EdgeType = typename Edges::value_type;

  REORDERING_BARRIER
  get_timer().start("local_kernelization_preprocess", 0);
  REORDERING_BARRIER
  auto min_cut_weights =
      compute_min_cut_weights(Span<EdgeType>(edges.data(), edges.size()),
                              vertex_range, [&vertex_range](const auto& edge) {
                                return is_true_local(edge, vertex_range);
                              });
  auto local_edges = get_local_edges(edges, vertex_range, min_cut_weights);
  mpi::MPIContext ctx;

  auto parents = parlay::sequence<VId>::from_function(
      vertex_range.n(), [&](size_t i) { return vertex_range.v_begin + i; });
  Span<EdgeType> all_edges_span(edges.data(), edges.size());
  Span<EdgeType> local_edges_span(local_edges.data(), local_edges.size());
  Span<std::atomic<EdgeIdWeight>> min_cut_weights_span(min_cut_weights.data(),
                                                       min_cut_weights.size());
  Span<VId> parents_span(parents.data(), parents.size());
  const std::size_t num_already_found_mst_edges = mst_edge_ids.size();
  mst_edge_ids.resize(num_already_found_mst_edges + vertex_range.n());
  REORDERING_BARRIER
  get_timer().stop("local_kernelization_preprocess", 0);
  get_timer().start("local_kernelization_boruvka", 0);
  REORDERING_BARRIER

  // const auto num_mst_edges =
  //     boruvka(vertex_range, local_edges_span, min_cut_weights_span, parents,
  //             mst_edge_ids.data() + num_already_found_mst_edges);
  const auto num_mst_edges = boruvka_filter(
      vertex_range, local_edges_span, min_cut_weights_span, parents_span,
      mst_edge_ids.data() + num_already_found_mst_edges);
  mst_edge_ids.resize(num_already_found_mst_edges + num_mst_edges);

  REORDERING_BARRIER
  get_timer().stop("local_kernelization_boruvka", 0);
  get_timer().start("local_kernelization_postprocessing1", 0);
  REORDERING_BARRIER
  dump(local_edges);
  int round = 0;
  constexpr bool compactify = false;
  get_timer().start("local_kernelization_postprocessing1_exchange", 0);
  DistributedGraph<typename Edges::value_type, compactify> graph(edges, round);
  auto name_newName_ghost_vertices_ =
      ExchangeRepresentativesPush_Sort::execute(graph, parents, round);
  get_timer().stop("local_kernelization_postprocessing1_exchange", 0);
  get_timer().start("local_kernelization_postprocessing1_relabel", 0);
  EdgeRenamer::rename_edges(graph, parents, name_newName_ghost_vertices_);
  get_timer().stop("local_kernelization_postprocessing1_relabel", 0);
  get_timer().start("local_kernelization_postprocessing1_remove", 0);
  EdgeProcessor::remove_self_loops(graph, round);
  // SEQ_EX(ctx, std::cout << vertex_range.v_begin << " " << vertex_range.v_end
  // << std::endl;);
  auto is_local = [&](const auto& edge) {
    const VId src = edge.get_src();
    const VId dst = edge.get_dst();
    const auto& v_begin = vertex_range.v_begin;
    const auto& v_end = vertex_range.v_end;
    return (v_begin <= src && src <= v_end) && (v_begin <= dst && dst <= v_end);
  };
  ParallelEdgeRemovalViaSampling::remove_duplicates(graph, is_local,
                                                    vertex_range.v_begin);
  get_timer().stop("local_kernelization_postprocessing1_remove", 0);
  get_timer().start("local_kernelization_postprocessing1_reorder", 0);
  reorder_splitted_edges(edges);
  EdgeProcessor::remove_duplicates(edges, round); // could potentially be improved - shouldn't be too bad
  get_timer().stop("local_kernelization_postprocessing1_reorder", 0);
  get_timer().stop("local_kernelization_postprocessing1", 0);
  return parents;
}

} // namespace internal_local_contraction
template <typename Edges, typename MstEdgeIds>
auto local_contraction(Edges&& edges, MstEdgeIds& mst_edge_ids,
                       double threshold = 0.1) {
  using namespace internal_local_contraction;
  mpi::MPIContext ctx;
  using EdgeType = typename std::decay_t<Edges>::value_type;
  // const VertexRange_ vertex_range(range.first, range.second + 1);
  const VertexRange_ vertex_range(edges.front().get_src(),
                                  edges.back().get_src() + 1);
  const std::size_t nb_local_edges =
      parlay::count_if(edges, [&](const auto& edge) {
        return is_true_local(edge, vertex_range);
      });

  const std::size_t nb_edges = edges.size();
  const std::size_t num_local_edges_complete =
      mpi::allreduce_sum(nb_local_edges, ctx);
  const std::size_t num_edges_complete = mpi::allreduce_sum(edges.size(), ctx);
  if (ctx.rank() == 0) {
    PRINT_VAR(num_local_edges_complete);
    PRINT_VAR(num_edges_complete);
    double ratio_local_edges =
        static_cast<double>(num_local_edges_complete) / num_edges_complete;
    PRINT_VAR(ratio_local_edges);
  }
  auto parents = parlay::sequence<VId>::from_function(
      vertex_range.n(), [&](size_t i) { return vertex_range.v_begin + i; });
  if ((num_local_edges_complete / static_cast<double>(num_edges_complete)) <=
      threshold) {
    if (ctx.rank() == 0) {
      std::cout << ctx.rank() << " do nothing" << std::endl;
    }
    return std::make_pair(std::move(parents), vertex_range);
  }
  parents = internal_local_contraction::execute_contraction(edges, vertex_range,
                                                            mst_edge_ids);
  // SEQ_EX(ctx, PRINT_VECTOR(edges); PRINT_CONTAINER_WITH_INDEX(parents););
  return std::make_pair(std::move(parents), vertex_range);
}

template <typename Edges, typename MstEdgeIds>
auto local_contraction(Edges&& edges, MstEdgeIds& mst_edge_ids,
                       ParentArray& parent_array, double threshold = 0.1) {

  const auto [local_parent_array, vertex_range] =
      local_contraction(std::forward<Edges>(edges), mst_edge_ids, threshold);

  REORDERING_BARRIER
  get_timer().start("local_kernelization_postprocessing2", 0);
  REORDERING_BARRIER

  non_init_vector<ParentArray::VertexParent> parent_updates(
      local_parent_array.size());

  hybridMST::parallel_for(0, parent_updates.size(), [&](const std::size_t& i) {
    parent_updates[i] = {i + vertex_range.v_begin, local_parent_array[i]};
  });
  auto filtered_updates = parlay::filter(parent_updates, [](const auto& entry) {
    return entry.index != entry.value;
  });
  parent_array.update(filtered_updates);
  REORDERING_BARRIER
  get_timer().stop("local_kernelization_postprocessing2", 0);
  REORDERING_BARRIER
  return local_parent_array;
}

template <typename Edges, typename MstEdgeIds>
auto remove_one_factors(Edges& edges, MstEdgeIds& mst_edge_ids) {
  using namespace internal_local_contraction;
  // asumption array is sorted
  mpi::MPIContext ctx;
  const std::size_t edge_offset = edges.front().get_edge_id();
  const VertexRange_ vertex_range(edges.front().get_src(),
                                  edges.back().get_src() + 1);
  non_init_vector<std::uint8_t> is_degree_one_vertex(edges.size());
  assign_initialize(is_degree_one_vertex,
                    [&](const auto& i) { return has_degree_one(i, edges); });
  Edge min_edge{edges.front().get_src(), edges.front().get_dst()};
  Edge max_edge{edges.back().get_src(), edges.back().get_dst()};
  VertexLocator_Split locator(min_edge, max_edge);
  auto filter = [&](const auto&, const std::size_t& i) {
    return !is_degree_one_vertex[i];
  };
  auto transformer = [&](const auto& edge, const std::size_t&) {
    const VId src = edge.get_src();
    const VId dst = edge.get_dst();
    return Edge{dst, src};
  };
  auto dst_computer = [&](const auto& edge, const std::size_t&) {
    const VId src = edge.get_src();
    const VId dst = edge.get_dst();
    return locator.get_min_pe(Edge{dst, src});
  };
  auto one_factors =
      mpi::alltoall_combined(edges, filter, transformer, dst_computer);
  auto& buffer = one_factors.buffer;
  ips4o::parallel::sort(buffer.begin(), buffer.end(), SrcDstOrder<Edge>{});
  auto is_less = [](const auto& lhs, const auto& rhs) {
    return std::make_tuple(lhs.get_src(), lhs.get_dst()) <
           std::make_tuple(rhs.get_src(), rhs.get_dst());
  };
  auto is_equal = [](const auto& lhs, const auto& rhs) {
    return std::make_tuple(lhs.get_src(), lhs.get_dst()) ==
           std::make_tuple(rhs.get_src(), rhs.get_dst());
  };

  std::size_t j = 0;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    for (; j < buffer.size() && is_less(buffer[j], edges[i]); ++j) {
    }
    if (j >= buffer.size())
      break;
    if (is_equal(buffer[j], edges[i])) {
      VId src = edges[i].get_src();
      VId dst = edges[i].get_dst();
      if (is_degree_one_vertex[i] && src < dst)
        mst_edge_ids.push_back(edges[i].get_edge_id());
      edges[i].set_dst(src);
    }
  }
  for (std::size_t i = 0; i < edges.size(); ++i) {
    if (!is_degree_one_vertex[i]) {
      continue;
    }
    const VId src = edges[i].get_src();
    const VId dst = edges[i].get_dst();
    if (src == dst)
      continue;
    mst_edge_ids.push_back(edges[i].get_edge_id());
    edges[i].set_dst(src);
  }
  EdgeProcessor::remove_self_loops(edges, 0);
  const bool is_sorted =
      std::is_sorted(edges.begin(), edges.end(),
                     SrcDstWeightOrder<typename Edges::value_type>{});
  SEQ_EX(ctx, PRINT_VAR(is_sorted););
}
} // namespace hybridMST
