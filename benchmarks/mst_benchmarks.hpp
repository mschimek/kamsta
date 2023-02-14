#pragma once

#include <iomanip>
#include <limits>
#include <random>
#include <type_traits>

#include "algorithms/base_case_mst_algos.hpp"
#include "util/allocators.hpp"
#include "util/benchmark_helpers.hpp"
#include "util/macros.hpp"

#include "definitions.hpp"

namespace hybridMST::benchmarks {

enum class Algorithm {
  hybridBoruvka,
  filter_hybridBoruvka,
  local_gbbs,
  local_gbbs_reimplementation,
  local_kruskal,
  local_kruskal_seq
};
enum class GraphType {
  GNM,
  RGG_2D,
  RGG_3D,
  RMAT,
  RHG,
  GRID_2D,
  GRID_3D,
  INFILE_UNWEIGHTED,
  INFILE_WEIGHTED
};

inline EnumMapper<GraphType> graphtypes{
    std::make_pair(GraphType::GNM, std::string("GNM")),
    std::make_pair(GraphType::RGG_2D, std::string("RGG_2D")),
    std::make_pair(GraphType::RGG_3D, std::string("RGG_3D")),
    std::make_pair(GraphType::RMAT, std::string("RMAT")),
    std::make_pair(GraphType::RHG, std::string("RHG")),
    std::make_pair(GraphType::GRID_2D, std::string("GRID_2D")),
    std::make_pair(GraphType::GRID_3D, std::string("GRID_3D")),
    std::make_pair(GraphType::INFILE_UNWEIGHTED,
                   std::string("INFILE_UNWEIGHTED")),
    std::make_pair(GraphType::INFILE_WEIGHTED, std::string("INFILE_WEIGHTED"))};

inline EnumMapper<Algorithm> algorithms{
    std::make_pair(Algorithm::hybridBoruvka, std::string("hybridBoruvka")),
    std::make_pair(Algorithm::filter_hybridBoruvka,
                   std::string("filter_hybridBoruvka")),
    std::make_pair(Algorithm::local_gbbs, std::string("local_gbbs")),
    std::make_pair(Algorithm::local_gbbs_reimplementation,
                   std::string("local_gbbs_reimplementation")),
    std::make_pair(Algorithm::local_kruskal, std::string("local_kruskal")),
    std::make_pair(Algorithm::local_kruskal_seq,
                   std::string("local_kruskal_seq"))};
inline EnumMapper<graphs::DistanceType> distance_types{
    std::make_pair(graphs::DistanceType::Random, std::string("RANDOM")),
    std::make_pair(graphs::DistanceType::Euclidean, std::string("EUCLIDEAN")),
    std::make_pair(graphs::DistanceType::SquaredEuclidean, std::string("SQUARED_EUCLIDEAN"))};

struct GraphParameters {
  std::string graphtype = graphtypes.get_string(GraphType::GNM);
  std::vector<std::string> infiles;
  std::size_t log_n = 8;
  std::size_t log_m = 12;
  double r = 0.025;
  double gamma = 3.0;
  bool is_weak_scaled_num_workers = false; // if weak scaling is enabled, n and
                                           // m are not global but per worker
  bool is_weak_scaled_num_mpi_procs =
      false; // if weak scaling is enabled, n and m are global but per mpi
             // process
  std::string distance_type = distance_types.get_string(graphs::DistanceType::Random);
  std::size_t max_edge_weight = 254;
};

struct AlgoParameters {
  std::string algo = algorithms.get_string(Algorithm::hybridBoruvka);
  std::size_t local_kernelization_level = 1;
  std::size_t filter_threshold = 4; // only relevant for filter approach
};

inline std::size_t
compute_weak_scaling_log_constant_kagen(const GraphParameters& params) {
  mpi::MPIContext ctx;
  if (params.is_weak_scaled_num_workers) {
    return std::log2(ctx.size() * ctx.threads_per_mpi_process());
  }
  if (params.is_weak_scaled_num_mpi_procs) {
    return std::log2(ctx.size());
  }
  return 0ull;
}

inline std::pair<std::size_t, std::size_t>
compute_numbers_rmat(const GraphParameters& params) {
  mpi::MPIContext ctx;
  const std::size_t log_mpi_procs = std::log2(ctx.size());
  if (params.is_weak_scaled_num_workers) {
    const std::size_t log_threads = std::log2(ctx.threads_per_mpi_process());
    return std::make_pair(params.log_n + log_mpi_procs + log_threads,
                          params.log_m + log_threads);
  }
  if (params.is_weak_scaled_num_mpi_procs) {
    return std::make_pair(params.log_n, params.log_m);
  }
  return std::make_pair(params.log_n, params.log_m - log_mpi_procs);
}

inline std::pair<WEdgeList14, VertexRange>
generate_graph(const GraphParameters& params) {
  mpi::MPIContext ctx;
  auto graphtype = graphtypes.get_enum(params.graphtype);
  if (params.max_edge_weight > 254 && ctx.rank() == 0) {
    std::cout << "edge weights are too big" << std::endl;
  }
  const auto distance_type = distance_types.get_enum(params.distance_type);
  const graphs::WeightGeneratorConfig<uint8_t> wgen_config{
      1, std::uint8_t(params.max_edge_weight), distance_type,
      std::size_t(ctx.rank())};

  switch (graphtype) {
  case GraphType::GNM: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);
    return convert(graphs::get_gnm(params.log_n + weak_scaling_log_constant,
                                   params.log_m + weak_scaling_log_constant,
                                   wgen_config));
  }
  case GraphType::RGG_2D: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);
    return convert(graphs::get_rgg2D(params.log_n + weak_scaling_log_constant,
                                     params.log_m + weak_scaling_log_constant,
                                     wgen_config));
  }
  case GraphType::RGG_3D: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);
    return convert(graphs::get_rgg3D(params.log_n + weak_scaling_log_constant,
                                     params.log_m + weak_scaling_log_constant,
                                     wgen_config));
  }
  case GraphType::RMAT: {
    const auto [log_n, log_m] = compute_numbers_rmat(params);
    return convert(graphs::get_rmat_edges_evenly_distributed(
        graphs::RMatParams{log_n, log_m}, wgen_config));
  }
  case GraphType::RHG: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);
    return convert(graphs::get_rhg_explicit_num_edges(
        params.log_n + weak_scaling_log_constant,
        params.log_m + weak_scaling_log_constant, params.gamma, wgen_config));
  }
  case GraphType::GRID_2D: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);

    const std::size_t log_x_weak_scaling_log_constant =
        weak_scaling_log_constant / 2 + weak_scaling_log_constant % 2;
    const std::size_t log_y_weak_scaling_log_constant =
        weak_scaling_log_constant / 2;
    const std::size_t log_x = params.log_n + log_x_weak_scaling_log_constant;
    const std::size_t log_y = params.log_n + log_y_weak_scaling_log_constant;
    const bool is_periodic = true;
    return convert(
        graphs::get_grid2D(log_x, log_y, 1.0, is_periodic, wgen_config));
  }
  case GraphType::GRID_3D: {
    const std::size_t weak_scaling_log_constant =
        compute_weak_scaling_log_constant_kagen(params);
    const std::size_t log_n = params.log_n + weak_scaling_log_constant;
    const bool is_periodic = true;
    return convert(
        graphs::get_grid3D(log_n, log_n, log_n, 1.0, is_periodic, wgen_config));
  }
  case GraphType::INFILE_UNWEIGHTED: {
    return convert(graphs::read_unweighted_graph(
        params.infiles.front(), graphs::GraphFormat::MatrixMarket));
  }
  case GraphType::INFILE_WEIGHTED: {
    using Infiles = decltype(params.infiles);
    return convert(
        graphs::read_weighted_binaries<std::vector<graphs::WEdge>, Infiles>(
            params.infiles, 4, 1));
  }
  // case GraphType::INFILE_WEIGHTED: {
  //   return hybridMST::read_unweighted_graph(params.infile); // Todo replace
  // }
  default: {
    return std::make_pair(WEdgeList14{},
                          VertexRange{VID_UNDEFINED, VID_UNDEFINED});
  }
  }
}

struct CmdParameters {
  GraphParameters graph_params;
  AlgoParameters algo_params;
  std::size_t iterations = 1;
  bool do_check = false;
  std::size_t debug_level = 0;
  std::size_t threads_per_mpi_process = 1;
  std::string outfile;
  std::string config_file =
      ""; // since MPI startup is very large on 65 536 PEs we try to run several
          // configs within one call to mpiexec. If a config file is given, the
          // specificed configurations will be executed
  friend std::ostream& operator<<(std::ostream& out,
                                  const CmdParameters& parameters) {
    const auto& infiles = parameters.graph_params.infiles;
    const std::string first_infile =
        infiles.empty() ? "/none" : infiles.front();
    const std::string pure_filename =
        first_infile.substr(first_infile.find_last_of("/") + 1);
    out << " log_n=" << parameters.graph_params.log_n;
    out << " log_m=" << parameters.graph_params.log_m;
    out << " radius=" << parameters.graph_params.r;
    out << " gamma=" << parameters.graph_params.gamma;
    out << " graphtype=" << parameters.graph_params.graphtype;
    out << " algo=" << parameters.algo_params.algo;
    out << " filename=" << pure_filename;
    out << " local_kernelization_level="
        << parameters.algo_params.local_kernelization_level;
    out << " filter_threshold=" << parameters.algo_params.filter_threshold;
    out << " threads_per_mpi_proc=" << parameters.threads_per_mpi_process;
    out << " nb_iterations=" << parameters.iterations;
    out << " is_weak_scaled_num_workers="
        << parameters.graph_params.is_weak_scaled_num_workers;
    out << " is_weak_scaled_num_mpi_procs="
        << parameters.graph_params.is_weak_scaled_num_mpi_procs;
    out << " distance_type=" << parameters.graph_params.distance_type;
    out << " max_edge_weight=" << parameters.graph_params.max_edge_weight;
    return out;
  }
};

inline void detailed_analysis(hybridMST::WEdgeList actual_mst,
                              const hybridMST::WEdgeList expected_mst) {
  using namespace hybridMST;
  std::cout << "#acutal_mst: " << actual_mst.size()
            << " #expected_mst: " << expected_mst.size() << std::endl;
  std::set<WEdge> actual_mst_set;
  for (const auto edge : actual_mst) {
    actual_mst_set.emplace(edge.src, edge.dst, edge.weight);
  }
  for (const auto edge : actual_mst) {
    WEdge rev{edge.dst, edge.src, edge.weight};
    if (actual_mst_set.count(rev) > 0) {
      std::cout << "rev: " << rev << " is contained!" << std::endl;
    }
  }

  WEdgeList actual_not_in_expected;
  WEdgeList expected_not_in_actual;
  for (const auto edge : actual_mst) {
    auto it = std::find_if(expected_mst.begin(), expected_mst.end(),
                           [&](const WEdge& expected_edge) {
                             return edge == WEdge{expected_edge.get_dst(),
                                                  expected_edge.get_src(),
                                                  expected_edge.get_weight()} ||
                                    expected_edge == edge;
                           });
    if (it == expected_mst.end())
      actual_not_in_expected.push_back(edge);
  }
  for (const auto edge : expected_mst) {
    auto it = std::find_if(
        actual_mst.begin(), actual_mst.end(), [&](const WEdge& actual_edge) {
          return edge == WEdge{actual_edge.get_dst(), actual_edge.get_src(),
                               actual_edge.get_weight()} ||
                 actual_edge == edge;
        });
    if (it == actual_mst.end())
      expected_not_in_actual.push_back(edge);
  }
  PRINT_VECTOR(actual_not_in_expected);
  PRINT_VECTOR(expected_not_in_actual);
}

template <typename InputEdges>
inline bool check(hybridMST::WEdgeList& actual_mst_edges,
                  const InputEdges& input) {
  hybridMST::mpi::MPIContext ctx;
  const std::uint64_t actual_local_sum = sum_edge_weights(actual_mst_edges);
  const std::uint64_t actual_sum =
      hybridMST::mpi::allreduce_sum(actual_local_sum);
  hybridMST::get_timer().start("check_computation");
  auto expected_mst = gather_mst(input);
  auto actual_mst = hybridMST::mpi::allgatherv(actual_mst_edges);

  std::sort(expected_mst.begin(), expected_mst.end(),
            hybridMST::SrcDstWeightOrder<typename InputEdges::value_type>{});
  std::sort(actual_mst.begin(), actual_mst.end(),
            hybridMST::SrcDstWeightOrder<hybridMST::WEdge>{});
  if (ctx.rank() == 0) {
    auto unique_it = std::unique(actual_mst.begin(), actual_mst.end());
    for (auto it = unique_it; it != actual_mst.end(); ++it) {
      std::cout << "duplicate edge in mst: " << *it << std::endl;
    }
  }
  // SEQ_EX(ctx, PRINT_VECTOR(expected_mst););
  hybridMST::get_timer().stop("check_computation");
  if (ctx.rank() == 0) {
    const std::uint64_t expected_sum = sum_edge_weights(expected_mst);
    std::cout << "actual weight mst: " << actual_sum
              << " expected_sum: " << expected_sum << std::endl;
    // detailed_analysis(actual_mst, expected_mst);
    return expected_sum == actual_sum;
  }

  return true;
}
} // namespace hybridMST::benchmarks
