#pragma once

#include "datastructures/distributed_graph.hpp"
#include "definitions.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/context.hpp"
#include "mpi/gather.hpp"
#include "util/utils.hpp"

namespace hybridMST {

struct WEdgeHash {
  std::size_t operator()(const WEdge& elem) const {
    return elem.get_src() ^ (elem.get_dst() << 5) + elem.get_weight();
  }
};
struct WEdgeEqual {
  std::size_t operator()(const WEdge& lhs, const WEdge& rhs) const {
    return lhs.get_src() == rhs.get_src() && lhs.get_dst() == rhs.get_dst() &&
           lhs.get_weight() == rhs.get_weight();
  }
};

template<typename Edges>
std::unordered_set<WEdge, WEdgeHash, WEdgeEqual> insert_in_set(const Edges& edges) {
  std::unordered_set<WEdge, WEdgeHash, WEdgeEqual> map;
  for (const auto& edge : edges) {
    map.emplace(edge.get_src(), edge.get_dst(), edge.get_weight());
  }
  return map;
}
template <typename OwnEdges, typename RecvEdges>
bool check_for_back_edge(const OwnEdges& own_edges,
                         const RecvEdges& recv_edges) {
  const auto set = insert_in_set(own_edges);
  bool everythink_ok = true;
  for (const auto& edge : recv_edges) {
    WEdge reverse_edge{edge.get_dst(), edge.get_src(), edge.get_weight()};
    auto it = set.find(reverse_edge);
    if (it == set.end()) {
      std::cout << "error: edge: " << edge << " has no back edge" << std::endl;
      everythink_ok = false;
    }
  }
  return everythink_ok;
}

template <typename Edges> void check_graph_consistency(const Edges& edges) {
  mpi::MPIContext ctx;
  using EdgeType = typename Edges::value_type;
  Edges non_const_edges = edges;
  const auto sum_edges = mpi::allreduce_sum(edges.size());
  int everythink_ok = true;
  if (sum_edges < (1000 * ctx.size())) {

    std::vector<EdgeType> all_edges(ctx.rank() == 0 ? sum_edges : 0);
    mpi::gatherv(non_const_edges.data(), non_const_edges.size(), 0,
                 all_edges.data(), ctx);
    if (ctx.rank() == 0) {
      everythink_ok = check_for_back_edge(all_edges, all_edges);
    }
  } else {
    DistributedGraph<EdgeType, true> graph(non_const_edges, 1111);
    auto filter = False_Predicate{};
    auto transformer = [&](const EdgeType& edge, const std::size_t&) {
      return edge;
    };
    auto dst_computer = [&](const EdgeType& edge, const std::size_t&) {
      return graph.locator().get_min_pe(Edge{edge.get_dst(), edge.get_src()});
    };
    auto back_edges =
        mpi::alltoall_combined(edges, filter, transformer, dst_computer);

    everythink_ok = check_for_back_edge(edges, back_edges.buffer);
  }
  everythink_ok = mpi::allreduce_min(everythink_ok, ctx);
  if(ctx.rank() == 0) {
    std::cout << "Edge have been checked with res: " << everythink_ok << std::endl;
  }
}

} // namespace hybridMST
