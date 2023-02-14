#include "catch2/catch.hpp"

#include <algorithms/hybrid_boruvka.hpp>
#include <tbb/global_control.h>
#include <unordered_map>
#include <util/graph_gen.hpp>

#include "datastructures/distributed_graph.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "test/utils.hpp"
#include "util/benchmark_helpers.hpp"

namespace hybridMST::tests {

void compute_components_sequentially(DistributedGraph& graph,
                                     const std::vector<uint8_t>& is_rep,
                                     const std::vector<VId>& min_edges,
                                     const std::vector<VId>& computed_reps) {
  // augment local_representative_infos with global vertex ids
  mpi::MPIContext ctx;

  REQUIRE(is_rep.size() == min_edges.size());
  std::vector<std::pair<VId, bool>> globalId_is_rep;
  std::vector<std::pair<VId, VId>> globalId_predecessor;
  // SEQ_EX(graph.ctx_, PRINT_VECTOR(is_rep); PRINT_VECTOR(min_edges););
  for (std::size_t i = 0; i < is_rep.size(); ++i) {
    const VId global_id = graph.get_global_id(i);
    globalId_is_rep.emplace_back(global_id, is_rep[i]);
    if (!is_rep[i]) {
      auto e = graph.edges()[min_edges[i]];
      globalId_predecessor.emplace_back(src(e), dst(e));
    }
  }
  std::sort(globalId_is_rep.begin(), globalId_is_rep.end());
  globalId_is_rep = mpi::allgatherv(globalId_is_rep, ctx);
  globalId_predecessor = mpi::allgatherv(globalId_predecessor, ctx);

  std::unordered_map<VId, bool> is_rep_global;
  auto predecessors = std::unordered_map<VId, VId>{};
  for (std::size_t i = 1; i < globalId_is_rep.size(); ++i) {
    const auto [prev_v, is_prev_rep] = globalId_is_rep[i - 1];
    const auto [v, is_v_rep] = globalId_is_rep[i];
    if (prev_v == v)
      REQUIRE(is_prev_rep == is_v_rep);
  }
  for (std::size_t i = 0; i < globalId_is_rep.size(); ++i) {
    const auto [v, is_v_rep] = globalId_is_rep[i];
    is_rep_global[v] = is_v_rep;
  }
  for (std::size_t i = 0; i < globalId_predecessor.size(); ++i)
    predecessors[globalId_predecessor[i].first] =
        globalId_predecessor[i].second;
  REQUIRE(predecessors.size() ==
          globalId_predecessor.size()); // split vertices are all representaties
  // SEQ_EX(graph.ctx_, PRINT_MAP(predecessors); PRINT_VECTOR(is_rep_global););
  for (const auto [org_v, org_is_rep] : is_rep_global) {
    VId v = org_v;
    bool is_rep = org_is_rep;
    while (!is_rep) {
      auto it_pred = predecessors.find(v);
      REQUIRE(it_pred != predecessors.end());
      v = it_pred->second;
      auto it_rep = is_rep_global.find(v);
      REQUIRE(it_rep != is_rep_global.end());
      is_rep = it_rep->second;
    }
    predecessors[org_v] = v;
  }
  // SEQ_EX(graph.ctx_, PRINT_MAP(predecessors););
  for (std::size_t i = 0; i < computed_reps.size(); ++i) {
    const VId v_global_id = graph.get_global_id(i);
    const VId v_rep = computed_reps[i];
    auto it = predecessors.find(v_global_id);
    REQUIRE(it != predecessors.end());
    REQUIRE(v_rep == it->second);
  }
}
void compute_reps_sequentially(DistributedGraph& graph,
                               const std::vector<VId>& min_edge_idxs,
                               const std::vector<uint8_t>& computed_is_rep) {
  mpi::MPIContext ctx;
  auto edge_copy = graph.edges();
  std::sort(edge_copy.begin(), edge_copy.end(), SrcWeightOrder<WEdge>{});

  std::vector<Edge> min_edges;
  std::vector<VId> undefined;
  std::vector<VId> vertices;
  //SEQ_EX(ctx, PRINT_VECTOR(min_edge_idxs););
  for (std::size_t i = 0; i < min_edge_idxs.size(); ++i) {
    vertices.push_back(graph.get_global_id(i));
    const auto idx = min_edge_idxs[i];
    if (!is_defined(idx))
      undefined.push_back(graph.get_global_id(i));
    else {
    const auto& edge = graph.edges()[idx];
    min_edges.emplace_back(src(edge), dst(edge));
    }
  }

  SEQ_EX(ctx, std::cout << "0" << std::endl;);
  std::vector<Edge> min_edges_global = mpi::allgatherv(min_edges);
  std::vector<VId> undefined_global = mpi::allgatherv(undefined);
  std::vector<VId> vertices_global = mpi::allgatherv(vertices);
  //SEQ_EX(ctx, PRINT_VECTOR(min_edges_global););

  SEQ_EX(ctx, std::cout << "1" << std::endl;);
  std::sort(vertices_global.begin(), vertices_global.end());
  std::set<VId> is_rep;
  for (std::size_t i = 1; i < vertices_global.size(); ++i) {
    const auto prev = vertices_global[i - 1];
    const auto cur = vertices_global[i];
    if (prev == cur)
      is_rep.insert(cur);
  }

  SEQ_EX(ctx, std::cout << "2" << std::endl;);
  for (const auto& undefined : undefined_global) {
    is_rep.insert(undefined);
  }
  std::unordered_map<VId, VId> src_dst;
  for (const auto& [src, dst] : min_edges_global) {
    src_dst[src] = dst;
  }
  SEQ_EX(ctx, std::cout << "3" << std::endl;);

  for (const auto& [src, dst] : min_edges_global) {
    if (is_rep.count(src) > 0)
      continue;
    if (is_rep.count(dst) > 0)
      continue;
    Edge local{src, dst};
    auto it = src_dst.find(dst);
    if (it == src_dst.end())
      std::cout << ctx.rank() << " " << local << "  strange" << std::endl;
    if (src != it->second)
      continue;
    Edge remote{dst, it->second};
    if (IsRepresentative_Push::is_src_root(local, remote))
      is_rep.insert(src);
  }
  SEQ_EX(ctx, std::cout << "4" << std::endl;);
  for (std::size_t i = 0; i < computed_is_rep.size(); ++i) {
    const VId v_global_id = graph.get_global_id(i);
    const bool is_rep_actual = computed_is_rep[i];
    const bool is_rep_expected = (is_rep.count(v_global_id) > 0);
    if (is_rep_actual != is_rep_expected) {
      std::cout << "globalId: " << v_global_id << " actual: " << is_rep_actual
                << " expected: " << is_rep_expected << std::endl;
    }
    REQUIRE(is_rep_actual == is_rep_expected);
  }
}

void test_compute_components(std::size_t n, std::size_t m) {
  hybridMST::mpi::MPIContext ctx;
  // PseudoGraph graph(edges, range);
  auto [edges, range] = get_gnm(n, m);
  DistributedGraph grap(range, edges);

  std::vector<std::atomic<EdgeIdWeight>> min_edges_par(grap.local_n());
  compute_min_weight_edges(grap, min_edges_par);

  std::vector<VId, default_init_allocator<VId>> min_edge_idxs(
      min_edges_par.size());
  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, min_edge_idxs.size()),
                    [&](tbb::blocked_range<std::size_t> r) {
                      for (std::size_t i = r.begin(); i < r.end(); ++i) {
                        min_edge_idxs[i] = min_edges_par[i].load().edge_id;
                        if (min_edge_idxs[i] == static_cast<uint32_t>(-1))
                          min_edge_idxs[i] = VID_UNDEFINED;
                      }
                    });
  const auto is_rep =
      IsRepresentative_Push::compute_representatives(min_edge_idxs, grap);
   const auto rep_local_vertices =
       ComputeRepresentative::compute_representatives(grap, is_rep,
                                                      min_edge_idxs);

  SEQ_EX(ctx, std::cout << "test" << std::endl;);

  compute_reps_sequentially(grap, vector_standard_alloc(min_edge_idxs), is_rep);
  compute_components_sequentially(grap, is_rep,
                                   vector_standard_alloc(min_edge_idxs),
                                   vector_standard_alloc(rep_local_vertices));
}
} // namespace hybridMST::tests

TEST_CASE("Boruvka Component 1", "[alltoall]") {
  hybridMST::tests::test_compute_components(10, 12);
  // hybridMST::tests::test_compute_components(15, 9);
}
TEST_CASE("Boruvka Component 1.2", "[alltoall]") {
  hybridMST::tests::test_compute_components(8, 10);
  // hybridMST::tests::test_compute_components(15, 9);
}
TEST_CASE("Boruvka Component 2", "[alltoall]") {
  hybridMST::tests::test_compute_components(5, 7);
  hybridMST::tests::test_compute_components(10, 15);
}
TEST_CASE("Boruvka Component 3", "[alltoall]") {
  hybridMST::tests::test_compute_components(17, 20);
  hybridMST::tests::test_compute_components(20, 22);
  hybridMST::tests::test_compute_components(22, 25);
}
