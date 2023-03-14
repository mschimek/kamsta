#pragma once

#include "definitions.hpp"
#include "shared_mem_parallel.hpp"
#include "util/utils.hpp"

namespace hybridMST {
struct MinimumEdge {
  static non_init_vector<LocalEdgeId> EdgeIdWeight_to_id(
      const non_init_vector<EdgeIdWeight>& edgeId_weights) {
    non_init_vector<LocalEdgeId> edgeIds(edgeId_weights.size());
    parallel_for(0, edgeId_weights.size(), [&](std::size_t i) {
      const auto& elem = edgeId_weights[i];
      edgeIds[i] = elem.edge_id;
    });
    return edgeIds;
  }
  static non_init_vector<LocalEdgeId> EdgeIdWeight_to_id(
      const std::vector<std::atomic<EdgeIdWeight>>& edgeId_weights) {
    non_init_vector<LocalEdgeId> edgeIds(edgeId_weights.size());
    parallel_for(0, edgeId_weights.size(), [&](std::size_t i) {
      const auto elem = edgeId_weights[i].load();
      edgeIds[i] = elem.edge_id;
    });
    return edgeIds;
  }

  static non_init_vector<LocalEdgeId> EdgeIdWeightDst_to_id(
      const std::vector<std::atomic<EdgeIdWeightDst>>& edgeId_weight_dsts) {
    non_init_vector<LocalEdgeId> edgeIds(edgeId_weight_dsts.size());
    parallel_for(0, edgeId_weight_dsts.size(), [&](std::size_t i) {
      const auto elem = edgeId_weight_dsts[i].load();
      edgeIds[i] = elem.edge_id;
    });
    return edgeIds;
  }
  static non_init_vector<LocalEdgeId> EdgeIdWeightDst_to_id(
      const non_init_vector<std::atomic<EdgeIdWeightDst>>& edgeId_weight_dsts) {
    non_init_vector<LocalEdgeId> edgeIds(edgeId_weight_dsts.size());
    parallel_for(0, edgeId_weight_dsts.size(), [&](std::size_t i) {
      const auto elem = edgeId_weight_dsts[i].load();
      edgeIds[i] = elem.edge_id;
    });
    return edgeIds;
  }
};

struct MinimumEdgeOpenMP : public MinimumEdge {
  template <typename Graph>
  static non_init_vector<LocalEdgeId> execute(const Graph& graph) {
    non_init_vector<std::atomic<EdgeIdWeightDst>> min_edges(graph.local_n());
    map(min_edges,
        [&](std::atomic<EdgeIdWeightDst>& atomic, const std::size_t&) {
          return std::atomic_init(
              &atomic, EdgeIdWeightDst{LOCAL_EDGEID_UNDEFINED, WEIGHT_INF,
                                       VID_UNDEFINED});
        });
    auto comp = [](const EdgeIdWeightDst& lhs, const EdgeIdWeightDst& rhs) {
      if (lhs.weight == rhs.weight) {
        return lhs.dst < rhs.dst;
      }
      return lhs.weight < rhs.weight;
    };

    parallel_for(0, graph.edges().size(), [&](std::size_t i) {
      const auto& elem = graph.edges()[i];
      const EdgeIdWeightDst id_weight{static_cast<uint32_t>(i),
                                      elem.get_weight(), elem.get_dst()};
      write_min(min_edges[graph.get_local_id(elem.get_src())], id_weight, comp);
    });
    return EdgeIdWeightDst_to_id(min_edges);
  }
};

struct MinimumEdgeSeq : public MinimumEdge {
  template <typename Graph>
  static non_init_vector<LocalEdgeId> execute(const Graph& graph) {
    non_init_vector<EdgeIdWeight> min_edges(graph.local_n());
    parallel_for(0, min_edges.size(), [&](std::size_t i) {
      min_edges[i].edge_id = LOCAL_EDGEID_UNDEFINED;
      min_edges[i].weight = WEIGHT_MAX;
    });

    get_timer().start("sequential_min");
    for (std::size_t i = 0; i < graph.edges().size(); ++i) {
      const auto& elem = graph.edges()[i];
      const auto& local_id = graph.get_local_id(elem.get_src());
      if (min_edges[local_id].weight > elem.get_weight()) {
        min_edges[local_id].edge_id = i;
        min_edges[local_id].weight = elem.get_weight();
      }
    }
    get_timer().stop("sequential_min");
    return EdgeIdWeight_to_id(min_edges);
  }
};
}  // namespace hybridMST
