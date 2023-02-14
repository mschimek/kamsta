#pragma once

#include <cstdint>
#include <mpi.h>
#include <vector>

#include "context.hpp"
#include "grid_communicators.hpp"
#include "mpi/context.hpp"
#include "mpi/grid_communicators.hpp"
#include "mpi/type_handling.hpp"
#include "type_handling.hpp"
#include "util/communication_volume_measurements.hpp"
#include "util/macros.hpp"

namespace hybridMST::mpi {

template <typename DataType>
inline std::vector<DataType> gather(const DataType& send_data, int32_t target,
                                    const MPIContext& ctx = MPIContext()) {
  std::vector<DataType> result(ctx.size());
  TypeMapper<DataType> tm;
  {
    const size_t recv_bytes =
        ctx.rank() == target ? sizeof(DataType) * ctx.size() : 0;
    get_communication_tracker().add_volume(sizeof(DataType), recv_bytes);
  }
  MPI_Gather(&send_data, 1, tm.get_mpi_datatype(), result.data(), 1,
             tm.get_mpi_datatype(), target, ctx.communicator());
  return result;
}

template <typename DataType>
inline void gatherv(DataType* send_data, int32_t send_count, int32_t target,
                    DataType* recv_buffer,
                    const MPIContext& ctx = MPIContext()) {
  std::vector<int32_t> receiving_sizes = gather(send_count, target, ctx);

  std::vector<std::int32_t> receiving_offsets(ctx.size(), 0);
  for (size_t i = 1; i < receiving_sizes.size(); ++i) {
    receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
  }

  TypeMapper<DataType> tm;
  {
    const std::size_t recv_bytes = ctx.rank() == target ?
        std::accumulate(receiving_sizes.begin(), receiving_sizes.end(), 0ull) *
        sizeof(DataType) : 0;
    const std::size_t sent_bytes = send_count * sizeof(DataType);
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }
  MPI_Gatherv(send_data, send_count, tm.get_mpi_datatype(), recv_buffer,
              receiving_sizes.data(), receiving_offsets.data(),
              tm.get_mpi_datatype(), target, ctx.communicator());
}

template <typename DataType,
          template <typename> typename Allocator = std::allocator>
inline std::vector<std::remove_cv_t<DataType>,
                   Allocator<std::remove_cv_t<DataType>>>
gatherv(DataType* send_data, int32_t send_count, int32_t target,
        const MPIContext& ctx = MPIContext()) {
  using DataType_ = std::remove_cv_t<DataType>;
  std::vector<int32_t> receiving_sizes = gather(send_count, target, ctx);

  std::vector<std::int32_t> receiving_offsets(ctx.size(), 0);
  for (size_t i = 1; i < receiving_sizes.size(); ++i)
    receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
  const std::size_t sum_sizes =
      receiving_offsets.back() + receiving_sizes.back();
  std::vector<DataType_, Allocator<DataType_>> recv_buffer(sum_sizes);
  {
    const std::size_t recv_bytes = target == ctx.rank() ?
        std::accumulate(receiving_sizes.begin(), receiving_sizes.end(), 0ull) *
        sizeof(DataType) : 0;
    const std::size_t sent_bytes = send_count * sizeof(DataType);
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }

  TypeMapper<DataType_> tm;
  MPI_Gatherv(send_data, send_count, tm.get_mpi_datatype(), recv_buffer.data(),
              receiving_sizes.data(), receiving_offsets.data(),
              tm.get_mpi_datatype(), target, ctx.communicator());
  return recv_buffer;
}
template <typename Container>
Container gatherv(const Container& send_data, int32_t target,
                  const MPIContext& ctx = MPIContext()) {
  using DataType = std::remove_cv_t<typename Container::value_type>;
  mpi::MPIContext world_ctx;

  std::vector<int32_t> receiving_sizes =
      gather(static_cast<int32_t>(send_data.size()), target, ctx);

  std::vector<std::int32_t> receiving_offsets(ctx.size(), 0);
  for (size_t i = 1; i < receiving_sizes.size(); ++i)
    receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
  const std::size_t sum_sizes =
      receiving_offsets.back() + receiving_sizes.back();
  Container recv_buffer(sum_sizes);
  {
    const std::size_t recv_bytes = ctx.rank() == target ?
        std::accumulate(receiving_sizes.begin(), receiving_sizes.end(), 0ull) *
        sizeof(DataType) : 0;
    const std::size_t sent_bytes = send_data.size() * sizeof(DataType);
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }
  TypeMapper<DataType> tm;
  MPI_Gatherv(send_data.data(), send_data.size(), tm.get_mpi_datatype(),
              recv_buffer.data(), receiving_sizes.data(),
              receiving_offsets.data(), tm.get_mpi_datatype(), target,
              ctx.communicator());
  return recv_buffer;
}
namespace two_level {
template <typename Container>
inline Container gatherv(Container& data, int32_t root) {
  using DataType = std::remove_cv_t<typename Container::value_type>;

  const auto& grid = get_power_two_grid_communicators();
  const int32_t root_in_column_comm = grid.column_id(root);
  const int32_t root_in_row_comm = grid.row_id(root);
  mpi::MPIContext ctx;
  auto column_wise_data =
      gatherv(data, root_in_column_comm, grid.get_col_ctx());
  if (root_in_column_comm != grid.get_col_ctx().rank()) {
    get_communication_tracker().add_volume(0,0);
    get_communication_tracker().add_volume(0,0);
    return Container{};
  }
  return gatherv(column_wise_data, root_in_row_comm, grid.get_row_ctx());
}
} // namespace two_level
} // namespace hybridMST::mpi
