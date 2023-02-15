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

#include "algorithms/algorithm_configurations.hpp"
#include "algorithms/base_case_mst_algos.hpp"
#include "algorithms/hybrid_boruvka.hpp"
#include "algorithms/hybrid_boruvka_edgefilter.hpp"
#include "datastructures/compression/difference_encoded_graph.hpp"
#include "datastructures/growt.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mst_benchmarks.hpp"
#include "util/benchmark_helpers.hpp"
#include "util/communication_volume_measurements.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

#include "build_config.hpp"

inline hybridMST::benchmarks::CmdParameters read_cmdline(int argc,
                                                         char* argv[]) {
  hybridMST::mpi::MPIContext ctx;
  hybridMST::benchmarks::CmdParameters params;
  std::size_t weak_scaling_level = 0;
  tlx::CmdlineParser cp;
  cp.set_description("hybrid mst computation");
  cp.set_author("Matthias Schimek");
  cp.add_stringlist('f', "infile", params.graph_params.infiles,
                    "path to infile to use for mst computation");
  cp.add_string("outfile", params.outfile,
                "path to file where measurements should be written");
  cp.add_string("configfile", params.config_file, "path to config file:");
  cp.add_size_t("log_n", params.graph_params.log_n,
                "log_2 of number of vertices");
  cp.add_size_t("log_m", params.graph_params.log_m, "log_2 of number of edges");
  cp.add_string('a', "algorithm", params.algo_params.algo,
                "algorithm to use for mst computation");
  cp.add_string('g', "graphtype", params.graph_params.graphtype,
                "graph to use as input "
                "<GNM|RMAT|RGG_2D|RGG_3D|RHG|GRID_2D|INFILE_WEIGHTED>");
  cp.add_size_t('i', "iterations", params.iterations,
                "number of iterations to run");
  cp.add_size_t('t', "threads", params.threads_per_mpi_process,
                "number of threads per mpi process. This number must match "
                "the number of OpenMP threads specificed via OMP_NUM_THREADS");
  cp.add_size_t("filter_threshold", params.algo_params.filter_threshold,
                "avg vertex degree for the base case in filter boruvka");
  cp.add_bool('c', "checks", params.do_check,
              "check the computed mst sequentially");
  cp.add_double('r', "radius", params.graph_params.r,
                "radius for random geometric graphs");
  cp.add_double("gamma", params.graph_params.gamma,
                "gamma value for random hyperbolic graphs");
  cp.add_bool("weak_scaling_workers",
              params.graph_params.is_weak_scaled_num_workers,
              "perform weak scaling on worker level (each used core)");
  cp.add_bool("print_input", params.print_input,
              "print the input (for debugging)");
  cp.add_string(
      "distance_type", params.graph_params.distance_type,
      "choose the distance type (RANDOM, EUCLIDEAN, SQUARED_EUCLIDEAN)");
  cp.add_size_t("max_edge_weight", params.graph_params.max_edge_weight,
                "maximum weight of edge");
  cp.add_size_t(
      "local_kernelization_level", params.algo_params.local_kernelization_level,
      "0 = do not use kernelization, 1 = contract local edges and vertices");
  cp.add_size_t("compression_level", params.algo_params.compression_level,
                "0 = do not use compression, 1 = use seven bit encoding");
  cp.add_size_t("weak_scaling_level", weak_scaling_level,
                "set weak scaling level: 0 - no weak scaling, 1 - weak "
                "scaling on worker level (each "
                "core used)(D), 2 - weak scaling on mpi proc level");

  cp.add_size_t("debug_level", params.debug_level,
                "set debug level: 0 - no debug output, 1 - debug output");

  if (!cp.process(argc, argv)) ctx.abort("problem reading cmd arguments");
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

#ifdef USE_EXPLICIT_INSTANTIATION
extern template hybridMST::WEdgeList hybridMST::boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_4_1,
                                     hybridMST::WEdgeId_4_1_7>&);
extern template hybridMST::WEdgeList hybridMST::boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_5_1,
                                     hybridMST::WEdgeId_6_1_7>&);
extern template hybridMST::WEdgeList hybridMST::boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_6_1,
                                     hybridMST::WEdgeId24>&);
extern template hybridMST::WEdgeList hybridMST::boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge, hybridMST::WEdgeId>&);

extern template hybridMST::WEdgeList hybridMST::filter_boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_4_1,
                                     hybridMST::WEdgeId_4_1_7>&);
extern template hybridMST::WEdgeList hybridMST::filter_boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_5_1,
                                     hybridMST::WEdgeId_6_1_7>&);
extern template hybridMST::WEdgeList hybridMST::filter_boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge_6_1,
                                     hybridMST::WEdgeId24>&);
extern template hybridMST::WEdgeList hybridMST::filter_boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge, hybridMST::WEdgeId>&);
#endif

template <typename InputEdges, typename WEdgeType, typename WEdgeIdType>
void run_experiments(InputEdges input_edges, hybridMST::VertexRange range,
                     const hybridMST::benchmarks::CmdParameters& params) {
  static_assert(
      std::is_same_v<InputEdges, typename std::vector<hybridMST::WEdge>>);
  hybridMST::mpi::MPIContext ctx;
  using InputWEdgeType = typename InputEdges::value_type;
  std::size_t edge_offset =
      hybridMST::VID_UNDEFINED;  // the edge offset is not really needed here
                                 // as we only use the compre
  const hybridMST::DifferenceEncodedGraph<InputWEdgeType, hybridMST::WEdgeId>
      compressed_input_graph(input_edges, ctx.threads_per_mpi_process(),
                             edge_offset, range);

  // only an additional check to ensure that the input is not modified
  {
    InputEdges input_edges_ = compressed_input_graph.get_WEdgeInSTDVector();
    if (input_edges_ != input_edges) {
      for (std::size_t i = 0; i < input_edges.size(); ++i) {
        if (!(input_edges_[i] == input_edges[i])) {
          std::cout << input_edges_[i] << " " << input_edges[i] << std::endl;
        }
      }
      PRINT_WARNING_AND_ABORT("input edges not equal! ... aborting now");
    }
    input_edges = std::move(input_edges_);
  }

  const auto algo =
      hybridMST::benchmarks::algorithms.get_enum(params.algo_params.algo);

  hybridMST::AlgorithmConfig<WEdgeType, WEdgeIdType> config;

  hybridMST::get_timer().reset();
  hybridMST::get_communication_tracker().reset();
  for (std::size_t i = 0; i < params.iterations; ++i) {
    input_edges = compressed_input_graph.get_WEdgeInSTDVector();
    hybridMST::WEdgeList mst_edges;

    config.local_preprocessing = static_cast<hybridMST::LocalPreprocessing>(
        params.algo_params.local_kernelization_level);
    config.compression = static_cast<hybridMST::Compression>(
        params.algo_params.compression_level);
    config.filter_threshold = params.algo_params.filter_threshold;

    if (params.print_input && i == 0) {
      SEQ_EX(ctx, PRINT_VECTOR(input_edges););
    }
    REORDERING_BARRIER
    hybridMST::get_timer().start("mst_complete", i);
    REORDERING_BARRIER

    switch (algo) {
      case hybridMST::benchmarks::Algorithm::hybridBoruvka: {
        mst_edges = hybridMST::boruvka(std::move(input_edges), config);
        break;
      }
      case hybridMST::benchmarks::Algorithm::filter_hybridBoruvka: {
        mst_edges = hybridMST::filter_boruvka(std::move(input_edges), config);
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
        hybridMST::mpi::allreduce_sum(compressed_input_graph.num_local_edges());
    const std::uint64_t num_vertices = hybridMST::mpi::allreduce_max(
        compressed_input_graph.get_range().second + 1);
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
      auto edges = compressed_input_graph.get_WEdgeList();
      if (!hybridMST::benchmarks::check(mst_edges, edges)) {
        sstream << "wrong result" << std::endl;
        if (ctx.rank() == 0) {
          std::cerr << "wrong result" << std::endl;
        }
        is_result_wrong = true;
      }
    }
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

    if (is_result_wrong) std::terminate();
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
  const bool use_small_edge_weights =
      params.graph_params.max_edge_weight <= 255u;
  const bool VIds_fit_in_32_bit_with_small_weights =
      n < (1ull << 32) && use_small_edge_weights;
  const bool VIds_fit_in_40_bit_with_small_weights =
      n < (1ull << 40) && use_small_edge_weights;
  const bool VIds_fit_in_48_bit_with_small_weights =
      n < (1ull << 48) && use_small_edge_weights;
  const bool VIds_fit_in_32_bit = n < (1ull << 32);
  if (VIds_fit_in_32_bit_with_small_weights) {
    using EdgeType = hybridMST::WEdge_4_1;
    hybridMST::print_on_root("used edge type: " +
                             hybridMST::get_edge_type_name<EdgeType>());
    run_experiments<decltype(edges), EdgeType, hybridMST::WEdgeId_4_1_7>(
        std::move(edges), range, params);
    return;
  }
  if (VIds_fit_in_40_bit_with_small_weights) {
    using EdgeType = hybridMST::WEdge_5_1;
    hybridMST::print_on_root("used edge type: " +
                             hybridMST::get_edge_type_name<EdgeType>());
    run_experiments<decltype(edges), EdgeType, hybridMST::WEdgeId_6_1_7>(
        std::move(edges), range, params);
    return;
  }
  if (VIds_fit_in_48_bit_with_small_weights) {
    using EdgeType = hybridMST::WEdge_6_1;
    hybridMST::print_on_root("used edge type: " +
                             hybridMST::get_edge_type_name<EdgeType>());
    run_experiments<decltype(edges), EdgeType, hybridMST::WEdgeId_6_1_7>(
        std::move(edges), range, params);
    return;
  }
  if (VIds_fit_in_32_bit) {
    using EdgeType = hybridMST::WEdge_4_4;
    hybridMST::print_on_root("used edge type: " +
                             hybridMST::get_edge_type_name<EdgeType>());
    run_experiments<decltype(edges), EdgeType, hybridMST::WEdgeId_4_4_8>(
        std::move(edges), range, params);
    return;
  }
  // default case
  using EdgeType = hybridMST::WEdge;
  hybridMST::print_on_root("used edge type: " +
                           hybridMST::get_edge_type_name<EdgeType>());
  run_experiments<decltype(edges), EdgeType, hybridMST::WEdgeId>(
      std::move(edges), range, params);

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
  for (const auto& config : configs) {
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
  auto start = std::chrono::steady_clock::now();
  bool is_rank_0 = false;
  hybridMST::mpi::MPIContext ctx;  // calls MPI_Init internally
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
