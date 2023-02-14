#include "catch2/catch.hpp"

#include "algorithms/hybrid_boruvka_re_computations.hpp"
#include "algorithms/local_contraction.hpp"
#include "util/graph_gen.hpp"

namespace hybridMST::tests {

struct PrevNameNewName {
  PrevNameNewName() = default;
  PrevNameNewName(VId prev_name_, VId new_name_)
      : prev_name{prev_name_}, new_name{new_name_} {}
  VId prev_name;
  VId new_name;
  friend std::ostream& operator<<(std::ostream& out,
                                  const PrevNameNewName& elem) {
    return out << "(" << elem.prev_name << ", " << elem.new_name << ")";
  }
};

template <typename EdgeType>
void rename_edges_globally(VertexRange_ range, const Span<VId> parents,
                           Span<EdgeType> edges) {
  hybridMST::mpi::MPIContext ctx;

  auto minmax = find_min_max(execution::parallel{}, edges);
  const auto& min_edge = edges[minmax.first];
  const auto& max_edge = edges[minmax.second];
  const Edge unweighted_min_edge{src(min_edge), dst(min_edge)};
  const Edge unweighted_max_edge{src(max_edge), dst(max_edge)};
  auto locator_split =
      VertexLocator_Split{unweighted_min_edge, unweighted_max_edge};
  std::vector<PrevNameNewName> prev_name_new_name;
  for (std::size_t i = 0; i < range.n(); ++i) {
    prev_name_new_name.emplace_back(range.v_begin + i, parents[i]);
  }
  prev_name_new_name = mpi::allgatherv(prev_name_new_name);
  std::unordered_map<VId, VId> map;
  for (const auto [prev_name, new_name] : prev_name_new_name) {
    map.emplace(prev_name, new_name);
  }
  for (auto& edge : edges) {
    const VId dst = dst_ref(edge);
    auto it = map.find(dst);
    dst_ref(edge) = it->second;
  }
}

inline bool check(const hybridMST::WEdgeList& actual_mst_edges,
                  const hybridMST::WEdgeList& input) {
  hybridMST::mpi::MPIContext ctx;
  const std::uint64_t actual_local_sum = sum_edge_weights(actual_mst_edges);
  const std::uint64_t actual_sum =
      hybridMST::mpi::allreduce_sum(actual_local_sum);
  const hybridMST::WEdgeList expected_mst = gather_mst(input);
  if (ctx.rank() == 0) {
    const std::uint64_t expected_sum = sum_edge_weights(expected_mst);
    REQUIRE(expected_sum == actual_sum);
  }
  return true;
}

void test_local_contraction(std::size_t log_n, std::size_t log_m) {
  auto [edges, range] = get_gnm(log_n, log_m);

  VertexRange_ vertex_range(range.first, range.second + 1);
  hybridMST::mpi::MPIContext ctx;
  std::vector<WEdgeId> augmented_edges(edges.size());
  for (LocalEdgeId i = 0; i != edges.size(); ++i) {
    const auto& edge = edges[i];
    augmented_edges[i] =
        WEdgeId{edge, EdgeIdDistribution::get_id(i, ctx.rank())};
  }
  std::vector<GlobalEdgeId> mst_edge_ids;
  auto parents = local_contraction(range, augmented_edges, mst_edge_ids);

  rename_edges_globally(vertex_range, Span(parents.data(), parents.size()),
                         Span(augmented_edges));
  auto gathered_edges =
      mpi::gatherv(augmented_edges.data(), augmented_edges.size(), 0, ctx);
  if (ctx.rank() == 0) {
    local_kruskal(1ull << log_n, gathered_edges, mst_edge_ids,
                  hybridMST::execution::parallel{});
  }
  const auto mst_edges = GetMstEdge::execute(edges, mst_edge_ids);
  check(mst_edges, edges);
}

} // namespace hybridMST::tests

using namespace hybridMST::tests;
TEST_CASE("LOCAL_CONTRACTION 1", "[general]") {
  test_local_contraction(10, 12);
  test_local_contraction(15, 17);
}
TEST_CASE("LOCAL_CONTRACTION 2", "[general]") {
  test_local_contraction(10, 8);
  test_local_contraction(15, 13);
}
TEST_CASE("LOCAL_CONTRACTION 3", "[general]") {
  test_local_contraction(18, 24);
} // namespace hybridMST::tests
