#include "catch2/catch.hpp"

#include "datastructures/distributed_graph_helper.hpp"
#include "datastructures/sparse_distributed_graph_helpers.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "test/utils.hpp"
#include "util/benchmark_helpers.hpp"
#include <mpi/broadcast.hpp>

namespace hybridMST::tests {

void splitter_setup() {
  using Interval = VertexLocator_Split::EdgeInterval;
  mpi::MPIContext ctx;
  if (ctx.size() < 5)
    return;
  std::vector<Interval> intervals{
      Interval{Edge(0, 5), Edge(10, 9)},
      Interval{Edge(10, 12), Edge(10, 100)},
      Interval{Edge(10, 100), Edge(11, 9)},
      Interval{Edge(12, 5), Edge(15, 9)},
      Interval{Edge(15, 9), Edge(18, 100)},
  };
  for (std::size_t i = intervals.size();
       i < static_cast<std::size_t>(ctx.size()); ++i) {
    intervals.push_back(Interval{});
  }
  VertexLocator_Split locator(intervals[ctx.rank()].min_edge,
                              intervals[ctx.rank()].max_edge);
  if (ctx.rank() == 0) {
    REQUIRE(!locator.is_v_min_split);
    REQUIRE(locator.is_v_max_split);
  }
  if (ctx.rank() == 1) {
    REQUIRE(locator.is_v_min_split);
    REQUIRE(locator.is_v_max_split);
  }
  if (ctx.rank() == 2) {
    REQUIRE(locator.is_v_min_split);
    REQUIRE(!locator.is_v_max_split);
  }
  if (ctx.rank() == 3) {
    REQUIRE(!locator.is_v_min_split);
    REQUIRE(locator.is_v_max_split);
  }
  if (ctx.rank() == 4) {
    REQUIRE(locator.is_v_min_split);
    REQUIRE(!locator.is_v_max_split);
  }
  {
    const auto req = locator.get_min_pe(Edge{10, 100});
    REQUIRE(req == 1);
  }
  {
    const auto req = locator.get_min_pe(10);
    REQUIRE(req == 0);
  }
  {
    const auto req = locator.get_max_pe(Edge{10, 100});
    REQUIRE(req == 2);
  }
  {
    const auto req = locator.get_max_pe(10);
    REQUIRE(req == 2);
  }
  {
    const auto req = locator.get_max_pe(13);
    REQUIRE(req == 3);
  }

  {
    const auto req = locator.get_min_pe(13);
    REQUIRE(req == 3);
  }

  {
    const auto req = locator.get_min_pe(Edge{14, 100});
    REQUIRE(req == 3);
  }

  {
    const auto req = locator.get_max_pe(Edge{14, 100});
    REQUIRE(req == 3);
  }
}
void sparse_splitter_setup() {
  using namespace sparse_graph;
  using Interval = VertexLocator_Split::EdgeInterval;
  mpi::MPIContext ctx;
  if (ctx.size() < 8)
    return;
  const VId max_vid = 500;
  std::vector<Interval> intervals{
      Interval{Edge(0, 5), Edge(10, 9)},
      Interval{Edge(10, 12), Edge(10, 100)},
      Interval{Edge(10, 100), Edge(11, 9)},
      Interval{Edge(12, 5), Edge(15, 9)},
      Interval{Edge(15, 9), Edge(18, 100)},
      Interval{Edge(18, 101), Edge(18, 106)},
      Interval{Edge(18, 106), Edge(18, 116)},
      Interval{Edge(18, 117), Edge(18, 200)},
      Interval{Edge(19, 5), Edge(20, max_vid)},
  };
  const Edge undefined_edge{VID_UNDEFINED, VID_UNDEFINED};
  std::vector<Edge> local_edges;
  const std::size_t num_local_edges = 10;
  for (std::size_t i = intervals.size();
       i < static_cast<std::size_t>(ctx.size()); ++i) {
    intervals.push_back(Interval{undefined_edge, undefined_edge});
  }
  const auto local_interval = intervals[ctx.rank()];
  if (!local_interval.is_empty()) {
    std::mt19937 gen(ctx.rank());
    // std::uniform_int_distribution<std::size_t> random_data(
    //     src(local_interval.min_edge), src(local_interval.max_edge));
    // for (std::size_t i = 0; i < num_local_edges; ++i) {
    //   local_edges.emplace_back(random_data(gen), random_data(gen));
    // }
    local_edges.emplace_back(0, 5);
    local_edges.emplace_back(1, 10);
    local_edges.emplace_back(10, 100);
    local_edges.emplace_back(18, 102);
    local_edges.emplace_back(18, 200);
    local_edges.emplace_back(20, max_vid);
  }
  sparse_graph::VertexLocator locator(local_interval.min_edge,
                                      local_interval.max_edge, local_edges);
  SEQ_EX(ctx, std::cout << locator.debug_print() << std::endl;);
  ctx.execute_in_order([&]() {
    for (auto& request : local_edges) {
      std::cout << request << ": " << locator.get_min_pe_and_split_info(request)
                << std::endl;
    }
  });

  // std::size_t src;
  // std::size_t dst;
  // int pe;
  // while (true) {
  //   if (ctx.rank() == 0) {
  //     std::cout << "give pe src dst" << std::endl;
  //     std::cin >> pe >> src >> dst;
  //   }
  //   pe = mpi::bcast(pe, 0, ctx);
  //   src = mpi::bcast(src, 0, ctx);
  //   dst = mpi::bcast(dst, 0, ctx);
  //   std::vector<Edge> edges;
  //   Edge e{src, dst};
  //   if (ctx.rank() == pe) {
  //     edges.push_back(e);
  //   }
  //   sparse_graph::VertexLocator locator(local_interval.min_edge,
  //                                       local_interval.max_edge,
  //                                       local_edges);
  //   if (ctx.rank() == pe) {
  //     std::cout << locator.get_min_pe_and_split_info(e) << std::endl;
  //   }
  // }
}
} // namespace hybridMST::tests

TEST_CASE("Split Locator", "[alltoall]") { hybridMST::tests::splitter_setup(); }
TEST_CASE("Sparse Split Locator", "[alltoall]") {
  hybridMST::tests::sparse_splitter_setup();
}
