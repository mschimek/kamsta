#pragma once

#include <type_traits>

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include <util/communication_volume_measurements.hpp>

namespace hybridMST::mpi {

template <typename DataType>
inline std::decay_t<DataType> exscan(const DataType& data, MPI_Op op,
                                     const MPIContext& ctx = MPIContext{}) {
  using DataType_ = std::decay_t<DataType>;
  TypeMapper<DataType> dtm;
  ctx.mpi_assert(dtm.is_builtin(), LOCATION_INFO + " no builtin type");
  DataType_ recv;
  {
    const std::size_t io_bytes = sizeof(DataType);
    get_communication_tracker().add_volume(io_bytes, io_bytes);
  }
  MPI_Exscan(&data, &recv, 1, dtm.get_mpi_datatype(), op, ctx.communicator());
  return recv;
}

template <typename DataType>
inline std::decay_t<DataType> exscan_sum(const DataType& data,
                                         const MPIContext& ctx = MPIContext{},
                                         DataType rank_zero = DataType{}) {
  auto result = exscan(data, MPI_SUM, ctx);
  return ctx.rank() == 0 ? rank_zero : result;
}
} // namespace hybridMST::mpi
