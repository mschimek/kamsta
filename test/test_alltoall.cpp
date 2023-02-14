#include "catch2/catch.hpp"

#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "util/benchmark_helpers.hpp"
#include "test/utils.hpp"


struct S {
  int i;
  char c;
  double d;
};

void test_alltoall(std::size_t n) {
  using namespace hybridMST::mpi;
  MPIContext ctx;
  const auto part_data = hybridMST::get_partitioned_data<int>(n, ctx);

  auto recv = hybridMST::mpi::twopass_alltoallv(
      part_data.data, hybridMST::False_Predicate{}, hybridMST::Identity{},
      [&](const int& elem, const std::size_t&) {
        return part_data.get_partition(elem);
      });
  recv = hybridMST::mpi::twopass_alltoallv(
      recv.buffer, hybridMST::False_Predicate{}, hybridMST::Identity{},
      [&](const int&, const std::size_t& idx) { return recv.get_pe(idx); });
  auto part_data_copy = part_data;
  REQUIRE(
      hybridMST::tests::contains_same_elements(recv.buffer, part_data.data));
}

void test_alltoall_openmp(std::size_t n, std::size_t nb_threads) {
  using namespace hybridMST::mpi;
  MPIContext ctx;
  const auto part_data = hybridMST::get_partitioned_data<int>(n, ctx);

  auto recv = hybridMST::mpi::twopass_alltoallv_openmp(
      part_data.data, hybridMST::False_Predicate{},
      hybridMST::Identity{},
      [&](const int& elem, const std::size_t&) {
        return part_data.get_partition(elem);
      },
      ctx.size(), nb_threads, 0);
  recv = hybridMST::mpi::twopass_alltoallv_openmp(
      recv.buffer, hybridMST::False_Predicate{}, hybridMST::Identity{},
      [&](const int&, const std::size_t& idx) { return recv.get_pe(idx); },
      ctx.size(), nb_threads, 0);
  auto part_data_copy = part_data;
  REQUIRE(hybridMST::tests::contains_same_elements(recv.buffer, part_data.data));
}

TEST_CASE("Alltoall Basic", "[alltoall]") {
  test_alltoall(100);
  test_alltoall(2);
  test_alltoall(1000);
  test_alltoall(10000);
  test_alltoall(100000);
}

TEST_CASE("Alltoall Openmp", "[alltoall]") {
  for (std::size_t i = 2; i <= 16; ++i) {
    test_alltoall_openmp(2, i);
    test_alltoall_openmp(21, i);
    test_alltoall_openmp(1000, i);
    test_alltoall_openmp(100000, i);
  }
}
