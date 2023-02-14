#include "catch2/catch.hpp"

#include "algorithms/distributed_partitioning.hpp"

namespace hybridMST::tests {
struct S {
  std::size_t a;
  std::size_t b;
  friend std::ostream& operator<<(std::ostream& out, const S& s) {
    return out << "(" << s.a << ", " << s.b << ")";
  }
};
struct Comparator {
  bool operator()(const S& lhs, const S& rhs) const {
    return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
  }
};
bool operator==(const S& lhs, const S& rhs) {
  return std::tie(lhs.a, lhs.b) == std::tie(rhs.a, rhs.b);
}
// using Comparator = std::less<std::size_t>;
// using S = std::size_t;
int i = 333333333333;
long int l = 22222222222;
double d = 333.333;
std::vector<S> generate_data(std::size_t n) {
  mpi::MPIContext ctx;
  std::mt19937 gen(ctx.rank());
  std::uniform_int_distribution<std::size_t> random_data(0, 100);
  std::vector<S> data(n);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = S{random_data(gen), random_data(gen)};
    // data[i] = S{random_data(gen)};
  }
  return data;
}

void test_partitioning(std::size_t local_n) {
  auto data = generate_data(local_n);
  auto all_data = mpi::allgatherv(data);
  std::sort(all_data.begin(), all_data.end(), Comparator{});

  auto partitioned_data = partition(data, Comparator{});
  std::sort(partitioned_data.begin(), partitioned_data.end(), Comparator{});
  const auto all_partitioned_data = mpi::allgatherv(partitioned_data);
  auto min_elements = mpi::allgather(partitioned_data.front());
  auto max_elements = mpi::allgather(partitioned_data.back());
  const bool is_min_sorted =
      std::is_sorted(min_elements.begin(), min_elements.end(), Comparator{});
  const bool is_max_sorted =
      std::is_sorted(max_elements.begin(), max_elements.end(), Comparator{});
  REQUIRE(is_min_sorted);
  REQUIRE(is_max_sorted);
  REQUIRE(all_partitioned_data.size() == all_data.size());
  for (std::size_t i = 0; i < all_partitioned_data.size(); ++i) {
    REQUIRE(all_partitioned_data[i] == all_data[i]);
  }
}

} // namespace hybridMST::tests
TEST_CASE("Distributed Partitioning 1", "[utils]") {
  hybridMST::tests::test_partitioning(10);
}

TEST_CASE("Distributed Partitioning 2", "[utils]") {
  hybridMST::tests::test_partitioning(100);
}

TEST_CASE("Distributed Partitioning 3", "[utils]") {
  hybridMST::tests::test_partitioning(1000);
}
