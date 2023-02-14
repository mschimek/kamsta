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

#include "alltoall_benchmarks.hpp"
#include "datastructures/growt.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mpi/twolevel_alltoall.hpp"
#include "util/benchmark_helpers.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

int get_keys(int key) {
  // volatile int i = 0;
  // for (int j = 0; j < 1000; ++j) i += j;
  return key;
}

std::vector<std::pair<int, int>> get_data(std::size_t n, std::size_t dist = 42,
                                          std::size_t p = 20) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, dist);
  std::uniform_int_distribution<> keys(0, p - 1);
  std::vector<std::pair<int, int>> vec(n);
  for (std::size_t i = 0; i < n; ++i) {
    vec[i] = std::make_pair(keys(gen), distrib(gen));
  }
  return vec;
}

void partition_seq_naive(const std::vector<std::pair<int, int>>& data,
                         std::size_t p,
                         std::vector<std::pair<int, int>>& output) {
  std::vector<std::vector<std::pair<int, int>>> meta_vec(p);
  output.clear();
  for (const auto& elem : data) {
    meta_vec[get_keys(elem.first)].push_back(elem);
  }
  for (const auto& vec : meta_vec) {
    std::copy(vec.begin(), vec.end(), std::back_inserter(output));
  }
}

struct GraphParameters {
  std::string graphtype = "GNM";
  std::string infile = "";
  std::size_t n = 10;
  std::size_t m = 15;
  double r = 0.0;
  bool is_weak_scaled =
      false; // if weak scaling is enabled, n and m are not global but per PE
};

inline hybridMST::benchmarks::CmdParameters read_cmdline(int argc,
                                                         char* argv[]) {
  hybridMST::mpi::MPIContext ctx;
  hybridMST::benchmarks::CmdParameters params;
  tlx::CmdlineParser cp;
  cp.set_description("hybrid mst computation");
  cp.set_author("Matthias Schimek");
  cp.add_size_t("log_number_items", params.alltoall_params.log_n,
                " log_2 of number of items to send");
  cp.add_double("density", params.alltoall_params.density,
                " density of message distribution (= ratio of PEs recv. data)");
  cp.add_string('a', "algorithm", params.alltoall_params.algo,
                " algorithm to use for mst computation");
  cp.add_size_t('i', "iterations", params.iterations,
                " nb of iterations to run");
  cp.add_size_t('t', "threads", params.threads_per_mpi_process,
                " nb of threads per mpi process");
  cp.add_bool('c', "checks", params.do_check, "do check afterwards");
  cp.add_size_t("debug_level", params.debug_level, "set debug level");

  if (!cp.process(argc, argv))
    ctx.abort("problem reading cmd arguments");
  ctx.set_threads_per_mpi_process(params.threads_per_mpi_process);
  return params;
}

template <typename T>
bool check(hybridMST::mpi::RecvMessages<T>& recv_msgs,
           hybridMST::benchmarks::AllToAllData<T> initial_data) {
  hybridMST::mpi::MPIContext ctx;
  auto recv_msgs2 = hybridMST::mpi::twopass_alltoallv_openmp_special(
       initial_data.get_data(), hybridMST::False_Predicate{},
          [](const int32_t elem, const std::size_t&) { return elem; },
          [&](const int32_t& /*elem*/, const std::size_t& idx) {
            return initial_data.get_dst(idx);
          },
      ctx.size(), ctx.threads_per_mpi_process());
  ips4o::parallel::sort(recv_msgs.buffer.begin(), recv_msgs.buffer.end());
  ips4o::parallel::sort(recv_msgs2.buffer.begin(), recv_msgs2.buffer.end());
  return recv_msgs2.buffer == recv_msgs.buffer;
}

void run_experiments(const hybridMST::benchmarks::CmdParameters& params) {
  const auto data = hybridMST::benchmarks::generate_data(params);
  hybridMST::mpi::MPIContext ctx;
  for (std::size_t i = 0; i < params.iterations; ++i) {
    hybridMST::mpi::RecvMessages<int32_t> recv_msgs;
    auto algo =
        hybridMST::benchmarks::algorithms.get_enum(params.alltoall_params.algo);
    hybridMST::get_timer().start("twopass_complete", i);

    SEQ_EX(ctx, PRINT_VECTOR(data.get_data()););
    switch (algo) {
    case hybridMST::benchmarks::Algorithm::Dense: {
      recv_msgs = hybridMST::mpi::twopass_alltoallv_openmp_special(
          data.get_data(), hybridMST::False_Predicate{},
          [](const int32_t elem, const std::size_t&) { return elem; },
          [&](const int32_t& /*elem*/, const std::size_t& idx) {
            return data.get_dst(idx);
          },
          ctx.size(), ctx.threads_per_mpi_process());
      break;
    }

    case hybridMST::benchmarks::Algorithm::TwoLevel: {
      recv_msgs = hybridMST::mpi::two_level_alltoall_extract(
          data.get_data(), hybridMST::False_Predicate{},
          [](const int32_t elem, const std::size_t&) { return elem; },
          [&](const int32_t& /*elem*/, const std::size_t& idx) {
            return data.get_dst(idx);
          });
      break;
    }
    }
    SEQ_EX(ctx, PRINT_VECTOR(recv_msgs.buffer););

    hybridMST::get_timer().stop("twopass_complete", i);
    std::stringstream setting;
    setting << " iteration=" << i;
    setting << " p=" << ctx.size();
    setting << params;
    hybridMST::get_timer().output(setting.str());
    hybridMST::get_timer().reset();
    if (params.do_check && !check(recv_msgs, data)) {
      std::cout << "wrong result" << std::endl;
      std::terminate();
    }
  }
}

int main(int argc, char* argv[]) {
  using namespace hybridMST;

  hybridMST::mpi::MPIContext ctx;
  const auto params = read_cmdline(argc, argv);
  run_experiments(params);
  mpi::MPIContext::finalize();
}
