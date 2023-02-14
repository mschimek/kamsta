#include "catch2/catch.hpp"

#include "mpi/twolevel_alltoall.hpp"
#include <util/utils.hpp>

namespace hybridMST::tests {
inline std::vector<int> generate_data(std::size_t n, int comm_size) {
  mpi::MPIContext ctx;
  std::mt19937 gen(ctx.rank() + 5);
  std::uniform_int_distribution<std::size_t> random_data(0, comm_size - 1);
  std::vector<int> data(n);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = random_data(gen);
  }
  return data;
}
template<typename T>
class TD;

template<typename T>
bool operator==(const std::vector<T>& lhs, const non_init_vector<T>& rhs) {
  if(lhs.size() != rhs.size())
    return false;
  for(std::size_t i = 0; i < lhs.size(); ++i)
    if(lhs[i] != rhs[i])
      return false;
  return true;
}

void test_twolevel_alltoall(std::size_t local_n) {
  mpi::MPIContext ctx;
  auto data = generate_data(local_n, ctx.size());
  //SEQ_EX(ctx, PRINT_VAR(grid_comms.get_row_ctx().size());
  //       PRINT_VAR(grid_comms.get_row_ctx().rank());
  //       PRINT_VAR(grid_comms.get_col_ctx().size());
  //       PRINT_VAR(grid_comms.get_col_ctx().rank()););
  //SEQ_EX(ctx, PRINT_VECTOR(data););

  auto filter = hybridMST::False_Predicate{};
  auto identity = [&](const auto& elem, std::size_t) { return elem; };
  auto destination = [&](const auto& elem, std::size_t) { return elem; };

  auto recv =
      hybridMST::mpi::two_level_alltoall(data, filter, identity, destination);
  auto extract_payload = [](const auto& elem, std::size_t) {
    return elem.payload;
  };
  auto extract_destination = [](const auto& elem, std::size_t) {
    return elem.get_sender();
  };
  auto reply = hybridMST::mpi::two_level_alltoall_extract(
      recv.buffer, filter, extract_payload, extract_destination);
  std::sort(data.begin(), data.end());
  std::sort(reply.buffer.begin(), reply.buffer.end());

  REQUIRE(data.size() == reply.buffer.size());
  for(std::size_t i = 0; i < data.size(); ++i)
    REQUIRE(data[i] == reply.buffer[i]);
  //SEQ_EX(ctx, PRINT_VECTOR(recv.buffer); PRINT_VECTOR(reply.buffer););
}
} // namespace hybridMST::tests

TEST_CASE("Twolevel Alltoall 1", "[utils]") {
  hybridMST::tests::test_twolevel_alltoall(10);
}

TEST_CASE("Twolevel Alltoall 2", "[utils]") {
  hybridMST::tests::test_twolevel_alltoall(1000);
}
