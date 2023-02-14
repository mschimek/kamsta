#pragma once

#include "definitions.hpp"
#include "mpi/context.hpp"
#include "util/utils.hpp"

namespace hybridMST {

struct EdgeProcessor {
  template <typename EdgeType>
  static void remove_self_loops(non_init_vector<EdgeType>& edges, int round) {

    mpi::MPIContext ctx;

    auto src_dst_equal = [](const EdgeType& lhs, const EdgeType& rhs) {
      return lhs.get_src() == rhs.get_src() && lhs.get_dst() == rhs.get_dst();
    };
    get_timer().start("remove_duplicate_edges_remove", round);

    auto filtered_edges = parlay::filter(
        edges, [&](const EdgeType& e) { return e.get_src() != e.get_dst(); });
    // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(filtered_edges););
    parallel_for(0, filtered_edges.size(),
                 [&](const auto& i) { edges[i] = filtered_edges[i]; });
    edges.erase(edges.begin() + filtered_edges.size(), edges.end());
    get_timer().stop("remove_duplicate_edges_remove", round);
    return;
  }

  template <typename Edges>
  static void remove_duplicates_impl(Edges& edges, int round) {
    using EdgeType = typename Edges::value_type;

    mpi::MPIContext ctx;

    auto src_dst_equal = [](const EdgeType& lhs, const EdgeType& rhs) {
      return lhs.get_src() == rhs.get_src() && lhs.get_dst() == rhs.get_dst();
    };
    get_timer().start("remove_duplicate_edges_sort", round);
    ips4o::parallel::sort(edges.begin(), edges.end(),
                          SrcDstWeightOrder<EdgeType>{});
    get_timer().stop("remove_duplicate_edges_sort", round);
    get_timer().start("remove_duplicate_edges_unify", round);

    // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(filtered_edges););
    if (edges.size() == 0) {
      edges.clear();
      get_timer().stop("remove_duplicate_edges_unify", round);
      get_timer().start("remove_duplicate_edges_erase", round);
      get_timer().stop("remove_duplicate_edges_erase", round);

      return;
    }
    non_init_vector<EdgeType> edges_tmp(edges.size());
    parallel_for(0, edges.size(),
                 [&](const auto& i) { edges_tmp[i] = edges[i]; });
    non_init_vector<std::uint8_t> is_survivor(edges.size());
    is_survivor[0] = 1;
    parallel_for(1, is_survivor.size(), [&](const auto& i) {
      const auto& prev = edges_tmp[i - 1];
      const auto& cur = edges_tmp[i];
      is_survivor[i] = !src_dst_equal(prev, cur);
    });
    auto num_remaining_edges = parlay::pack_into(edges_tmp, is_survivor, edges);
    get_timer().stop("remove_duplicate_edges_unify", round);
    get_timer().start("remove_duplicate_edges_erase", round);
    edges.erase(edges.begin() + num_remaining_edges, edges.end());
    get_timer().stop("remove_duplicate_edges_erase", round);
  }

  template <typename Graph>
  static void remove_self_loops(Graph& graph, int round) {
    auto& edges = graph.edges();
    remove_self_loops(edges, round);
  }
  template <typename EdgeType>
  static void remove_duplicates(non_init_vector<EdgeType>& edges, int round) {
    remove_duplicates_impl(edges, round);
    edges.shrink_to_fit();
  }

  template <typename Graph>
  static void remove_duplicates(Graph& graph, int round) {
    auto& edges = graph.edges();
    remove_duplicates_impl(edges, round);
  }
};

/// Removing parallel edges (u,v) keeping only the lightest among these turns
/// out to be a bottleneck for graphs with high locality. We therefore, propose
/// a sampling based parallel edge removal algorithm:
/// 1. Determine a pivot weight to partition the edges in two sets E_< and E_>
/// with E_< << E_>.
/// 2. Remove duplicates in E_< via sorting and store the remaining edges in a
/// set.
/// 3. Scan over E_> removing all edges that are already contained in the set.
/// 4. Remove duplicates in the remaining edges via sorting.
/// Note: preliminary experiments indicate that it is crucial to keep the set
/// structure small enough to fit into the cache. Otherwise plain sorting seems
/// to be faster.
struct ParallelEdgeRemovalViaSampling {
  ///@brief For local edges (u,v) for which u and v can be represented with 32
  /// bits this functions returns (u << 32 | v)
  template <typename EdgeType>
  static VId get_combined_id(EdgeType& local_edge, VId v_min) {
    VId combined_value = local_edge.get_src() - v_min;
    combined_value <<= 32;
    combined_value |= (local_edge.get_dst() - v_min);
    return combined_value;
  }

  template <typename Graph> static Weight determine_pivot(const Graph& graph, const double ratio = 0.05) {
    using EdgeType = typename Graph::EdgeType;
    if (graph.edges().empty())
      return 1;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> distrib(0, graph.edges().size() - 1);
    const std::size_t sample_size =
        std::max(std::size_t(1000), std::size_t(graph.edges().size() * 0.005));
    non_init_vector<EdgeType> sample(sample_size);
    for (std::size_t i = 0; i < sample_size; ++i) {
      const auto idx = sample[i] = graph.edges()[distrib(gen)];
    }
    ips4o::parallel::sort(sample.begin(), sample.end(),
                          WeightOrder<EdgeType>{});
    const Weight pivot = sample[ratio * sample_size].get_weight();
    return pivot;
  }

  template <typename Graph, typename IsLocal>
  static void remove_duplicates(Graph& graph, IsLocal& is_local, VId v_min) {
    using EdgeType = typename Graph::EdgeType;
    Weight pivot = determine_pivot(graph);
    auto initial_filter = [&](const auto& edge) {
      const auto& w = edge.get_weight();
      return (is_local(edge) & (w <= pivot));
    };
    auto sample_edges = parlay::filter(graph.edges(), initial_filter);
    const auto initial_sample_size = sample_edges.size();
    EdgeProcessor::remove_duplicates_impl(sample_edges, 0);
    std::unordered_set<VId> set;
    for (const auto& sample_edge : sample_edges) {
      const auto combined_id = get_combined_id(sample_edge, v_min);
      set.insert(combined_id);
    }
    auto filter = [&](const auto& edge) {
      const auto combined_id = get_combined_id(edge, v_min);
      return !is_local(edge) || (set.find(combined_id) == set.end());
    };
    auto edges_not_in_sample = parlay::filter(graph.edges(), filter);
    const auto initial_edge_size = graph.edges().size();
    non_init_vector<typename Graph::EdgeType> edges(edges_not_in_sample.size() +
                                                    sample_edges.size());
    parallel_for(0, sample_edges.size(),
                 [&](const auto& i) { edges[i] = sample_edges[i]; });
    parallel_for(0, edges_not_in_sample.size(), [&](const auto& i) {
      edges[i + sample_edges.size()] = edges_not_in_sample[i];
    });
    graph.edges() = std::move(edges);
    dump(edges);
    EdgeProcessor::remove_duplicates(graph, 0);
  }
};
} // namespace hybridMST
