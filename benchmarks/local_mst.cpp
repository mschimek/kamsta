#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include <omp.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "ips4o/ips4o.hpp"
#include "tlx/cmdline_parser.hpp"

#include "algorithms/base_case_mst_algos.hpp"
#include "algorithms/gbbs_reimplementation.hpp"
#include "datastructures/growt.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mst_benchmarks.hpp"
#include "util/benchmark_helpers.hpp"
#include "io/graph_gen.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

inline hybridMST::benchmarks::CmdParameters read_cmdline(int argc,
                                                         char* argv[]) {
  hybridMST::mpi::MPIContext ctx;
  hybridMST::benchmarks::CmdParameters params;
  tlx::CmdlineParser cp;
  cp.set_description("hybrid mst computation");
  cp.set_author("Matthias Schimek");
  cp.add_size_t("log_n", params.graph_params.log_n,
                " log_2 of number of vertices");
  cp.add_size_t("log_m", params.graph_params.log_m,
                " log_2 of number of edges");
  cp.add_string('a', "algorithm", params.algo_params.algo,
                " algorithm to use for mst computation");
  cp.add_string('g', "graphtype", params.graph_params.graphtype,
                " graph to use as input");
  cp.add_string('f', "infile", params.graph_params.infile,
                " infile to use for mst computation");
  cp.add_size_t('i', "iterations", params.iterations,
                " nb of iterations to run");
  cp.add_size_t('t', "threads", params.threads_per_mpi_process,
                " nb of threads per mpi process");
  cp.add_bool('c', "checks", params.do_check, "do check afterwards");
  cp.add_bool("weak_scaling", params.do_check, "perform weak scaling");
  cp.add_size_t("debug_level", params.debug_level, "set debug level");

  if (!cp.process(argc, argv))
    ctx.abort("problem reading cmd arguments");
  ctx.set_threads_per_mpi_process(
      static_cast<int>(params.threads_per_mpi_process));
  omp_set_num_threads(static_cast<int>(params.threads_per_mpi_process));
  if (params.debug_level > 0) {
    hybridMST::get_timer().set_debug_output_enablement(true);
  }
  return params;
}

template <typename EdgeType, typename Container>
std::vector<hybridMST::WEdge> get_edges(const std::vector<EdgeType>& edges,
                                        Container& ids) {
  using namespace hybridMST;
  std::vector<hybridMST::WEdge> res_edges;
  for (const auto& id : ids) {
    const auto& edge = edges[id];
    res_edges.emplace_back(src(edge), dst(edge), weight_ref(edge));
  }
  return res_edges;
}

template <typename Container>
inline bool check(const Container& mst_edge_ids,
                  const hybridMST::WEdgeList& input) {
  hybridMST::mpi::MPIContext ctx;
  const auto actual_mst_edges = get_edges(input, mst_edge_ids);
  const std::uint64_t actual_sum = sum_edge_weights(actual_mst_edges);
  hybridMST::get_timer().start("check_computation");
  const hybridMST::WEdgeList expected_mst = gather_mst(input);
  hybridMST::get_timer().stop("check_computation");
  // SEQ_EX(ctx, PRINT_VECTOR(expected_mst););
  if (ctx.rank() == 0) {
    const std::uint64_t expected_sum = sum_edge_weights(expected_mst);
    std::cout << "actual weight mst: " << actual_sum
              << " expected_sum: " << expected_sum << std::endl;
    return expected_sum == actual_sum;
  }
  return true;
}

template <typename Container>
std::vector<hybridMST::WEdgeId> wedge_to_wedgeId(const Container& w_edges) {
  std::vector<hybridMST::WEdgeId> w_edge_ids(w_edges.size());
  for (std::size_t i = 0; i < w_edges.size(); ++i) {
    const auto& edge = w_edges[i];
    w_edge_ids[i] =
        hybridMST::WEdgeId(src(edge), dst(edge), weight_ref(edge), i);
  }
  return w_edge_ids;
}

void run_experiments(const hybridMST::benchmarks::CmdParameters& params) {

  hybridMST::get_timer().start("gen");
  auto [edges, range] =
      hybridMST::benchmarks::generate_graph(params.graph_params);
  hybridMST::get_timer().stop("gen");
  hybridMST::mpi::MPIContext ctx;
  ips4o::parallel::sort(edges.begin(), edges.end(),
                        hybridMST::SrcDstWeightOrder<hybridMST::WEdge>{});
  const auto algo =
      hybridMST::benchmarks::algorithms.get_enum(params.algo_params.algo);
  for (std::size_t i = 0; i < params.iterations; ++i) {
    auto w_edge_ids = wedge_to_wedgeId(edges);
    // PRINT_VECTOR(w_edge_ids);
    const std::size_t n = range.second - range.first + 1;
    hybridMST::get_timer().start("mst_complete");

    std::vector<hybridMST::GlobalEdgeId> mst_edge_ids;
    mst_edge_ids.reserve(
        1ull << params.graph_params
                    .log_n); // should not clutter up performance measurement
                             // any further (and could be done better)
    switch (algo) {
    case hybridMST::benchmarks::Algorithm::local_gbbs: {
      hybridMST::gbbs_mst(n, w_edge_ids, mst_edge_ids);
      break;
    }
    case hybridMST::benchmarks::Algorithm::local_gbbs_reimplementation: {
      hybridMST::gbbs_reimplementation(n, w_edge_ids, mst_edge_ids);
      break;
    }
    case hybridMST::benchmarks::Algorithm::local_kruskal: {
      hybridMST::local_kruskal(n, w_edge_ids, mst_edge_ids,
                               hybridMST::execution::parallel{});
      break;
    }
    case hybridMST::benchmarks::Algorithm::local_kruskal_seq: {
      hybridMST::local_kruskal(n, w_edge_ids, mst_edge_ids,
                               hybridMST::execution::sequential{});
      break;
    }
    default:
      std::cout << "nothing chosen" << std::endl;
      std::cout << hybridMST::benchmarks::algorithms.get_string(algo)
                << std::endl;
    }
    hybridMST::get_timer().stop("mst_complete");
    std::stringstream setting;
    setting << " iteration=" << i;
    setting << " p=" << ctx.size();
    setting << params;
    if (params.do_check && !check(mst_edge_ids, edges)) {
      std::cout << "wrong result" << std::endl;
      std::terminate();
    }
    hybridMST::get_timer().output(setting.str());
    hybridMST::get_timer().reset();
  }
}

int main(int argc, char* argv[]) {
  using namespace hybridMST;

  hybridMST::mpi::MPIContext ctx;
  const auto params = read_cmdline(argc, argv);
  run_experiments(params);
  mpi::MPIContext::finalize();
}
