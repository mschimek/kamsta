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

#include "backward.hpp"
#include "ips4o/ips4o.hpp"
#include "read_config_file.hpp"
#include "tlx/cmdline_parser.hpp"

//#include "algorithms/boruvka_based_on_external_subfunctions.hpp"
#include "algorithms/base_case_mst_algos.hpp"
#include "algorithms/hybrid_boruvka.hpp"
#include "algorithms/hybrid_boruvka_edgefilter.hpp"
#include "datastructures/compressed_graph.hpp"
#include "datastructures/compression/difference_encoded_graph.hpp"
#include "datastructures/growt.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mst_benchmarks.hpp"
#include "util/benchmark_helpers.hpp"
#include "util/communication_volume_measurements.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

inline hybridMST::benchmarks::CmdParameters read_cmdline(int argc,
                                                         char* argv[]) {
  hybridMST::mpi::MPIContext ctx;
  hybridMST::benchmarks::CmdParameters params;
  std::size_t weak_scaling_level = 0;
  tlx::CmdlineParser cp;
  cp.set_description("hybrid mst computation");
  cp.set_author("Matthias Schimek");
  cp.add_stringlist('f', "infile", params.graph_params.infiles,
                    " infile to use for mst computation");
  cp.add_string("outfile", params.outfile,
                " path to file where measurements should be written");
  cp.add_string("configfile", params.config_file, " path to config file");
  cp.add_size_t("log_n", params.graph_params.log_n,
                " log_2 of number of vertices");
  cp.add_size_t("log_m", params.graph_params.log_m,
                " log_2 of number of edges");
  cp.add_string('a', "algorithm", params.algo_params.algo,
                " algorithm to use for mst computation");
  cp.add_string('g', "graphtype", params.graph_params.graphtype,
                " graph to use as input");
  cp.add_size_t('i', "iterations", params.iterations,
                " nb of iterations to run");
  cp.add_size_t('t', "threads", params.threads_per_mpi_process,
                " nb of threads per mpi process");
  cp.add_size_t("filter_threshold", params.algo_params.filter_threshold,
                " nb of threads per mpi process");
  cp.add_bool('c', "checks", params.do_check, "do check afterwards");
  cp.add_double('r', "radius", params.graph_params.r,
                "radius for random geometric graphs");
  cp.add_double("gamma", params.graph_params.gamma,
                "gamma value for random hyperbolic graphs");
  cp.add_bool("weak_scaling_workers",
              params.graph_params.is_weak_scaled_num_workers,
              "perform weak scaling on worker level (each used core)");
  cp.add_string(
      "distance_type", params.graph_params.distance_type,
      "choose the distance type (RANDOM, EUCLIDEAN, SQUARED_EUCLIDEAN)");
  cp.add_size_t("max_edge_weight", params.graph_params.max_edge_weight,
                "maximum weight of edge");
  cp.add_size_t(
      "local_kernelization_level", params.algo_params.local_kernelization_level,
      "0 = do not use kernelization, 1 = contract local edges and vertice");
  cp.add_size_t("weak_scaling_level", weak_scaling_level,
                "0 - no weak scaling, 1 - weak scaling on worker level (each \
                used core), 2 - weak scaling on mpi proc level");

  cp.add_size_t("debug_level", params.debug_level, "set debug level");

  if (!cp.process(argc, argv))
    ctx.abort("problem reading cmd arguments");
  switch (weak_scaling_level) {
  case 0:
    params.graph_params.is_weak_scaled_num_workers = false,
    params.graph_params.is_weak_scaled_num_mpi_procs = false;
    break;
  case 1:
    params.graph_params.is_weak_scaled_num_workers = true,
    params.graph_params.is_weak_scaled_num_mpi_procs = false;
    break;
  case 2:
    params.graph_params.is_weak_scaled_num_workers = false,
    params.graph_params.is_weak_scaled_num_mpi_procs = true;
    break;
  default:
    ctx.abort("problem with weak scaling argument");
  }
  ctx.set_threads_per_mpi_process(
      static_cast<int>(params.threads_per_mpi_process));
  if (params.debug_level > 0) {
    hybridMST::get_timer().set_debug_output_enablement(true);
  }
  return params;
}

template <typename CompressedGraph>
void run_experiments(const hybridMST::benchmarks::CmdParameters& params,
                     CompressedGraph& compressed_graph) {
  hybridMST::mpi::MPIContext ctx;
  const auto algo =
      hybridMST::benchmarks::algorithms.get_enum(params.algo_params.algo);
  const auto local_kernelization_level =
      params.algo_params.local_kernelization_level;
  const auto filter_threshold = params.algo_params.filter_threshold;
  hybridMST::get_timer().reset();
  hybridMST::get_communication_tracker().reset();
  for (std::size_t i = 0; i < params.iterations; ++i) {
    hybridMST::WEdgeList mst_edges;
    hybridMST::get_timer().start("mst_complete", i);
    REORDERING_BARRIER
    switch (algo) {
    case hybridMST::benchmarks::Algorithm::hybridBoruvka: {
      mst_edges =
          hybridMST::boruvka(compressed_graph, compressed_graph.get_range(),
                             local_kernelization_level);
      break;
    }
    case hybridMST::benchmarks::Algorithm::filter_hybridBoruvka: {
      mst_edges = hybridMST::filter_boruvka(
          compressed_graph, compressed_graph.get_range(),
          local_kernelization_level, filter_threshold);
      break;
    }
    default: {
      ctx.abort("algorithm not available in the benchmark");
      break;
    }
    }
    REORDERING_BARRIER
    hybridMST::get_timer().stop("mst_complete", i);
    const std::uint64_t sum_weights =
        hybridMST::mpi::allreduce_sum(sum_edge_weights(mst_edges));
    const std::uint64_t num_mst_edges =
        hybridMST::mpi::allreduce_sum(mst_edges.size());
    const std::uint64_t num_edges =
        hybridMST::mpi::allreduce_sum(compressed_graph.num_local_edges());
    const std::uint64_t num_vertices =
        hybridMST::mpi::allreduce_max(compressed_graph.get_range().second + 1);
    std::stringstream sstream;
    if (ctx.rank() == 0)
      sstream << " run finished; weight MSF:" << sum_weights
              << " #mst_edges: " << num_mst_edges << " #edges: " << num_edges
              << " #vertices: " << num_vertices << std::endl;
    std::stringstream setting;
    setting << " iteration=" << i;
    setting << " p=" << ctx.size();
    setting << params;
    bool is_result_wrong = false;
    if (params.do_check) {
      auto edges = compressed_graph.get_WEdgeList();
      if (!hybridMST::benchmarks::check(mst_edges, edges)) {
        sstream << "wrong result" << std::endl;
        is_result_wrong = true;
      }
    }
    auto comm_volume = hybridMST::get_communication_tracker()
                           .collect(); // has to be done before timer collection
    sstream << hybridMST::get_timer().output(setting.str());
    sstream << hybridMST::get_communication_tracker().output();
    const std::string content_to_print = sstream.str();

    if (ctx.rank() == 0) {
      std::cout << content_to_print << std::endl;
      if (!params.outfile.empty()) {
        std::ofstream out(params.outfile, std::ios::app);
        out << content_to_print;
        out.close();
      }
    }

    if (is_result_wrong)
      std::terminate();
    hybridMST::get_timer().reset();
    hybridMST::get_communication_tracker().reset();
  }
}

void run_experiments(const hybridMST::benchmarks::CmdParameters& params) {
  hybridMST::mpi::MPIContext ctx;
  hybridMST::get_timer().start("gen");
  auto [edges, range] =
      hybridMST::benchmarks::generate_graph(params.graph_params);
  hybridMST::get_timer().stop("gen");
  // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(edges););
  const std::size_t n = hybridMST::mpi::allreduce_max(range.second) + 1;
  const std::size_t edge_offset =
      hybridMST::mpi::exscan_sum(edges.size(), ctx, 0ul);
  const bool VIds_fit_in_32_bit = n < (1ull << 32);
  const bool VIds_fit_in_40_bit = n < (1ull << 40);
  const bool VIds_fit_in_48_bit = n < (1ull << 48);
  if (VIds_fit_in_32_bit) {
    // hybridMST::wait_for_user("compressed graph");
    hybridMST::DifferenceEncodedGraph<hybridMST::WEdge10, hybridMST::WEdgeId16>
        compressed_graph(edges, ctx.threads_per_mpi_process(), edge_offset,
                         range);
    // PRINT_VAR(compressed_graph.get_range());
    // hybridMST::verify_compressed_construction<
    //     decltype(edges), hybridMST::WEdge10, hybridMST::WEdgeId16>(edges,
    //                                                                range);
    // hybridMST::wait_for_user("compressed graph");
    hybridMST::dump(edges);
    // hybridMST::wait_for_user("after dump graph");
    run_experiments(params, compressed_graph);
    return;
  }
  if (VIds_fit_in_40_bit) {
    hybridMST::DifferenceEncodedGraph<hybridMST::WEdge12, hybridMST::WEdgeId20>
        compressed_graph(edges, ctx.threads_per_mpi_process(), edge_offset,
                         range);
    { auto dump = std::move(edges); }
    run_experiments(params, compressed_graph);
    return;
  }
  //if (VIds_fit_in_48_bit) {
  //  hybridMST::DifferenceEncodedGraph<hybridMST::WEdge14, hybridMST::WEdgeId24>
  //      compressed_graph(edges, ctx.threads_per_mpi_process(), edge_offset,
  //                       range);
  //  { auto dump = std::move(edges); }
  //  run_experiments(params, compressed_graph);
  //  return;
  //}
  //hybridMST::DifferenceEncodedGraph<hybridMST::WEdge, hybridMST::WEdgeId>
  //    compressed_graph(edges, ctx.threads_per_mpi_process(), edge_offset,
  //                     range);
  //{ auto dump = std::move(edges); }
  //run_experiments(params, compressed_graph);
  return;
}

void run_experiments_main(hybridMST::benchmarks::CmdParameters params) {
  if (params.config_file.empty()) {
    // normal procedure
    return run_experiments(params);
  }
  auto configs = hybridMST::benchmarks::parse_config_file(params.config_file);
  hybridMST::mpi::MPIContext ctx;
  if (ctx.rank() == 0) {
    std::cout << configs.size() << std::endl;
  }
  for (const auto config : configs) {
    if (ctx.rank() == 0) {
      std::cout << config.graphtype << std::endl;
    }
    params.graph_params = config;
    run_experiments(params);
  }
}

void run_experiments_verification(
    hybridMST::benchmarks::CmdParameters& params) {

  hybridMST::get_timer().start("gen");
  hybridMST::get_timer().stop("gen");
  hybridMST::mpi::MPIContext ctx;
  const auto algo =
      hybridMST::benchmarks::algorithms.get_enum(params.algo_params.algo);
  std::size_t log_n_start = params.graph_params.log_n;
  std::size_t log_m_start = params.graph_params.log_m;
  std::size_t max_edge_weight_start = params.graph_params.max_edge_weight;
  for (std::size_t log_n = log_n_start; log_n < log_n_start + 10; ++log_n) {
    for (std::size_t log_m = log_m_start;
         log_m < std::min(2 * log_n - 1, log_m_start + 10); ++log_m) {
      for (std::size_t max_edge_weight = max_edge_weight_start;
           max_edge_weight < 25; ++max_edge_weight) {
        params.graph_params.log_n = log_n;
        params.graph_params.log_m = log_m;
        params.graph_params.max_edge_weight = max_edge_weight;
        run_experiments(params);
      }
    }
  }
}
int main(int argc, char* argv[]) {
  using namespace hybridMST;

  auto [a, b] = std::make_pair(1, 2);
  auto start = std::chrono::steady_clock::now();
  bool is_rank_0 = false;
  hybridMST::mpi::MPIContext ctx;
  const auto params = read_cmdline(argc, argv);
  const double start_mpi = MPI_Wtime();
  if (ctx.rank() == 0) {
    is_rank_0 = true;
    std::cout << "MPI_Initialize" << std::endl;
  }
  {
    backward::MPIErrorHandler mpi_error_handler(MPI_COMM_WORLD);
    // enable backward for non-MPI failures
    backward::SignalHandling sh;

    auto params = read_cmdline(argc, argv);
    // run_experiments_verification(params);
    run_experiments_main(params);
  }
  const double end_mpi = MPI_Wtime();
  mpi::MPIContext::finalize();
  auto end = std::chrono::steady_clock::now();
  if (is_rank_0) {
    std::cout << "Time between MPI_Init() - MPI_Finalize(): "
              << end_mpi - start_mpi << std::endl;
    std::cout << "Overall time: "
              << (std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                        start)
                      .count() /
                  static_cast<double>(1000'000))
              << std::endl;
  }
}
