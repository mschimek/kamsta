#pragma once

#include "mst_benchmarks.hpp"
#include "tlx/cmdline_parser.hpp"
#include <fstream>

namespace hybridMST::benchmarks {
// from https://stackoverflow.com/a/46931770
inline std::vector<std::string> split(std::string s, std::string delimiter) {
  std::size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }
  res.push_back(s.substr(pos_start));
  return res;
}

inline std::vector<const char*>
emulate_argv(const std::vector<std::string>& args) {
  std::vector<const char*> argv;
  argv.push_back(args.front().data()); // emulate filename
  for (const auto& str : args) {
    argv.push_back(str.data());
  }
  return argv;
}

inline std::vector<GraphParameters>
parse_config_file(const std::string& path_config_file) {
  mpi::MPIContext ctx;
  std::vector<GraphParameters> params_list;
  auto config_file = std::ifstream(path_config_file);
  std::string line;
  while (std::getline(config_file, line)) {
    auto splitted = split(line, " ");
    const auto argv = emulate_argv(splitted);
    GraphParameters params;
    tlx::CmdlineParser cp;
    std::size_t weak_scaling_level = 1;

    cp.add_string("graphtype", params.graphtype, "");
    cp.add_size_t("log_n", params.log_n, "");
    cp.add_size_t("log_m", params.log_m, "");
    cp.add_size_t("log_m", params.log_m, "");
    cp.add_double("gamma", params.gamma, "");
    cp.add_string("distance_type", params.distance_type, "");
    cp.add_size_t("max_edge_weight", params.max_edge_weight, "");
    cp.add_size_t("weak_scaling_level", weak_scaling_level, "");
    switch (weak_scaling_level) {
    case 0:
      params.is_weak_scaled_num_workers = false,
      params.is_weak_scaled_num_mpi_procs = false;
      break;
    case 1:
      params.is_weak_scaled_num_workers = true,
      params.is_weak_scaled_num_mpi_procs = false;
      break;
    case 2:
      params.is_weak_scaled_num_workers = false,
      params.is_weak_scaled_num_mpi_procs = true;
      break;
    default:
      ctx.abort("problem with weak scaling argument");
    }
    cp.process(argv.size(), argv.data());
    params_list.push_back(params);
  }
  return params_list;
}
} // namespace hybridMST::benchmarks
