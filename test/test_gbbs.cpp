#include <iostream>
#include <vector>

#include "algorithms/base_case_mst_algos.hpp"
#include "definitions.hpp"
#include "util/utils.hpp"
#include "gbbs-fork/benchmarks/MinimumSpanningForest/Boruvka/interface.hpp"

std::vector<gbbs::WEdgeId> read_edges_gbbs(const std::string& filename) {
  std::ifstream in(filename);
  std::vector<gbbs::WEdgeId> edges;
  gbbs::uintE src, dst, w;
  std::size_t global_id;
  while (in >> src >> dst >> w >> global_id) {
    edges.emplace_back(src, dst, w, global_id);
  }
  return edges;
}

template <typename EdgeType, typename Container>
std::vector<hybridMST::WEdge> get_edges(
    const std::vector<EdgeType>& edges,
    Container& ids) {
  using namespace hybridMST;
  std::unordered_map<hybridMST::GlobalEdgeId, std::size_t> edge_id_idx;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    edge_id_idx.emplace(edges[i].global_id, i);
  }
  std::vector<hybridMST::WEdge> res_edges;
  for (const auto& id : ids) {
    const auto it = edge_id_idx.find(id);
    if (it == edge_id_idx.end()) {
      std::cout << "Id: " << id << " not found" << std::endl;
      std::terminate();
    }
    const auto& edge = edges[it->second];
    res_edges.emplace_back(src(edge), dst(edge), weight_ref(edge));
  }
  return res_edges;
}

int main() {
  std::string filename;
  std::cin >> filename;
  auto edges = read_edges_gbbs(filename);
  auto cpy = edges;
  auto cpy2 = edges;
  const std::size_t n = edges.back().src + 1;
  auto gbbs_ids = compute_minimum_spanning_forest(n, edges);

  std::vector<hybridMST::GlobalEdgeId> kruskal_edge_id;
  auto kruskal_input = hybridMST::convert_vertex_ids_to_8_byte(cpy);
  hybridMST::local_kruskal(n, kruskal_input, kruskal_edge_id);
  const auto kruskal_edges = get_edges(cpy2, kruskal_edge_id);
  const auto gbbs_edges = get_edges(cpy2, gbbs_ids);
  const auto kruskal_weight = hybridMST::sum_edge_weights(kruskal_edges);
  const auto gbbs_weight = hybridMST::sum_edge_weights(gbbs_edges);
  std::cout << "kruskal: " << kruskal_weight << " gbbs: " << gbbs_weight << std::endl;
}
