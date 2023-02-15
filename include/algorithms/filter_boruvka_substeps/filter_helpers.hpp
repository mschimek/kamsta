#pragma once

#include "parlay/hash_table.h"

#include "algorithms/distributed_partitioning.hpp"
#include "datastructures/distributed_parent_array.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "util/edge_checker.hpp"

namespace hybridMST {
class LabelCache {
public:
  LabelCache(std::size_t n) : map_(n * 1.1) {}
  template <typename Edges>
  void rename_edges(Edges& edges, const Weight& threshold_weight) {
    parallel_for(0, edges.size(), [&](std::size_t i) {
      auto& edge = edges[i];
      VId src_label = edge.get_src();
      VId dst_label = edge.get_dst();
      if (edge.get_weight() <= threshold_weight) {
        src_label = 0;
        dst_label = 0;
        edge.set_src(src_label);
        edge.set_dst(dst_label);
        return;
      }
      auto it_src_label = growt::find(map_, src_label);
      if (it_src_label != map_.end())
        src_label = (*it_src_label).second;
      auto it_dst_label = growt::find(map_, dst_label);
      if (it_dst_label != map_.end())
        dst_label = (*it_dst_label).second;
      edge.set_src(src_label);
      edge.set_dst(dst_label);
    });
  }
  void update_cache(const VId& key, const VId& value) {
    map_.insert_or_assign(key + 1, value);
  }

private:
  growt::GlobalVIdMap<VId> map_;
};
struct RetrieveNewLabels {
  template <typename Container>
  static auto execute(int round, Container& edges,
                      const ParentArray& parent_array,
                      Weight threshold_weight) {
    get_timer().start("retrieve_new_labels_init_init", round);
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        2 * edges.size(), parlay::hash_numeric<VId>{});
    get_timer().stop("retrieve_new_labels_init_init", round);
    get_timer().start("retrieve_new_labels_filter_init", round);
    parallel_for(0, edges.size(), [&](std::size_t i) {
      auto& edge = edges[i];
      if (edge.get_weight() <= threshold_weight) {
        return;
      }
      table.insert(edge.get_src());
      table.insert(edge.get_dst());
    });
    auto entries = table.entries();
    mpi::MPIContext ctx;
    get_timer().stop("retrieve_new_labels_filter_init", round);

    get_timer().add("retrieve_new_label_send_count", round, entries.size(),
                    {Timer::DatapointsOperation::ID});

    get_timer().start("retrieve_new_labels_get_init", round);
    auto res = parent_array.get_parents(entries, ParentArray::InVector{});
    get_timer().stop("retrieve_new_labels_get_init", round);
    // get_timer().add("retrieve_new_label_recv_count", round, entries.size(),
    //                 {Timer::DatapointsOperation::ID});
    return res;
  }
};

template <typename EdgeType>
void rename_edges(int round, Span<EdgeType> edges,
                  const ParentArray& parent_array, Weight threshold_weight,
                  LabelCache& cache) {
  mpi::MPIContext ctx;
  get_timer().start("retrieve_new_labels_init", round);
  auto v_parent_v =
      RetrieveNewLabels::execute(round, edges, parent_array, threshold_weight);
  LabelCache cache_(v_parent_v.size());
  get_timer().stop("retrieve_new_labels_init", round);
  get_timer().start("update_cache_init", round);
  parallel_for(0, v_parent_v.size(), [&](std::size_t i) {
    const auto& [v, parent_v] = v_parent_v[i];
    cache_.update_cache(v, parent_v);
  });
  get_timer().stop("update_cache_init", round);
  get_timer().start("rename_labels_init", round);
  cache_.rename_edges(edges, threshold_weight);
  get_timer().stop("rename_labels_init", round);
}

template <typename EdgeType>
non_init_vector<EdgeType> partition_edges(non_init_vector<EdgeType>& edges,
                                          Weight& pivot, bool& stop_recursion) {

  mpi::MPIContext ctx;
  auto local_samples = get_samples(edges, 8);
  const auto comp = WeightOrder<EdgeType>{};
  pivot = select_pivot(local_samples, comp);
  const auto is_light = [&](const auto& edge) {
    return edge.get_weight() <= pivot;
  };
  const auto is_heavy = [&](const auto& edge) {
    return edge.get_weight() > pivot;
  };

  const std::size_t num_edges = edges.size();
  const std::size_t num_light_edges = parlay::count_if(edges, is_light);
  non_init_vector<EdgeType> light_edges(num_light_edges);
  non_init_vector<EdgeType> heavy_edges(edges.size() - num_light_edges);
  parlay::filter_into(edges, light_edges, is_light);
  parlay::filter_into(edges, heavy_edges, is_heavy);
  edges = std::move(heavy_edges);
  // non_init_vector<EdgeType> heavy_edges(edges.size() - num_light_edges);
  // parlay::filter_into(edges, heavy_edges, [&](const EdgeType& edge) { return
  // weight_ref(edge) > pivot; }); edges = heavy_edges;
  std::size_t total_num_light_edges = mpi::allreduce_sum(num_light_edges, ctx);
  std::size_t total_num_edges = mpi::allreduce_sum(num_edges, ctx);
  if (total_num_edges == 0) {
    stop_recursion = true;
  } else {
    double balance_ratio =
        total_num_light_edges / static_cast<double>(total_num_edges);
    stop_recursion = balance_ratio > 0.85;
  }
  constexpr bool perform_check = false;
  if (perform_check) {
    check_graph_consistency(edges);
  }
  return light_edges;
}

} // namespace hybridMST
