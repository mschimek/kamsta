
#include "../include/mpi/alltoall.hpp"
#include "../include/mpi/context.hpp"
#include "../include/util/timer.hpp"
#include <omp.h>
#include <random>
#include <sstream>
#include <vector>

std::vector<int> gen_data(std::size_t n, int seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> distrib(0, 1000);
  std::vector<int> vec(n);
  for (std::size_t i = 0; i < n; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}
int main() {

  hybridMST::mpi::MPIContext ctx;

  const auto data = gen_data(200'000, ctx.rank());
  std::vector<int> splitters(ctx.size());
  for (std::size_t i = 0; i < splitters.size(); ++i) {
    splitters[i] = 10 * i;
  }
  splitters.back() = std::numeric_limits<int>::max();
  auto filter = [](const int, const int) { return false; };
  auto transform = [](const int elem, const int) { return elem; };
  auto dst = [&](const int elem, const int) {
    auto it = std::lower_bound(splitters.begin(), splitters.end(), elem);
    return std::distance(splitters.begin(), it);
  };
  for (std::size_t i = 0; i < 10; ++i) {
    ctx.barrier();

    auto res = hybridMST::mpi::twopass_alltoallv_openmp_special(
        data, filter, transform, dst, ctx.size(), 8);
    const auto red_res = hybridMST::mpi::allreduce_min(res.buffer.size());
    if (ctx.rank() == 0) {
      std::cout << red_res << std::endl;
    }

    std::stringstream setting;
    setting << " iteration=" << i;
    setting << " p=" << ctx.size();
    hybridMST::get_timer().output(setting.str());
    hybridMST::get_timer().reset();
  }
  hybridMST::mpi::MPIContext::finalize();
}
