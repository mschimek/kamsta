#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <type_traits>
#include <vector>

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include "util/communication_volume_measurements.hpp"

namespace hybridMST::mpi {

static inline bool allreduce_and(bool send_data,
                                 const MPIContext& ctx = MPIContext()) {
  bool receive_data;
  MPI_Allreduce(&send_data, &receive_data, 1, MPI_C_BOOL, MPI_LAND,
                ctx.communicator());
  return receive_data;
}

template <typename T>
using remove_cvr_t =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;
template <typename DataType>
inline remove_cvr_t<DataType>
allreduce_max(const DataType& send_data, const MPIContext& ctx = MPIContext()) {
  static_assert(std::is_arithmetic<DataType>(),
                "Only arithmetic types are allowed for allreduce_max.");
  using DataType_ = std::decay_t<DataType>;
  DataType_ receive_data;
  TypeMapper<DataType_> tm;
  {
    const std::size_t io_bytes = sizeof(DataType);
    get_communication_tracker().add_volume(io_bytes, io_bytes);
  }
  MPI_Allreduce(&send_data, &receive_data, 1u, tm.get_mpi_datatype(), MPI_MAX,
                ctx.communicator());
  return receive_data;
}

template <typename DataType>
inline remove_cvr_t<DataType>
allreduce_min(const DataType& send_data, const MPIContext& ctx = MPIContext()) {
  static_assert(std::is_arithmetic<DataType>(),
                "Only arithmetic types are allowed for allreduce_max.");
  using DataType_ = std::decay_t<DataType>;
  DataType_ receive_data;
  TypeMapper<DataType_> tm;
  {
    const std::size_t io_bytes = sizeof(DataType);
    get_communication_tracker().add_volume(io_bytes, io_bytes);
  }
  MPI_Allreduce(&send_data, &receive_data, 1u, tm.get_mpi_datatype(), MPI_MIN,
                ctx.communicator());
  return receive_data;
}

template <typename DataType>
inline remove_cvr_t<DataType>
allreduce_sum(const DataType& send_data, const MPIContext& ctx = MPIContext()) {
  static_assert(std::is_arithmetic<DataType>(),
                "Only arithmetic types are allowed for allreduce_max.");
  using DataType_ = std::decay_t<DataType>;
  DataType_ receive_data;
  TypeMapper<DataType_> tm;
  {
    const std::size_t io_bytes = sizeof(DataType);
    get_communication_tracker().add_volume(io_bytes, io_bytes);
  }
  MPI_Allreduce(&send_data, &receive_data, 1u, tm.get_mpi_datatype(), MPI_SUM,
                ctx.communicator());
  return receive_data;
}

template <int is_commutative, typename Op, typename T> struct CustomFunction {
  CustomFunction() {
    MPI_Op_create(CustomFunction<is_commutative, Op, T>::execute,
                  is_commutative, &mpi_op);
  }
  static void execute(void* invec, void* inoutvec, int* len,
                      MPI_Datatype* /*datatype*/) {
    T* invec_ = static_cast<T*>(invec);
    T* inoutvec_ = static_cast<T*>(inoutvec);
    Op op{};
    std::transform(invec_, invec_ + *len, inoutvec_, inoutvec_, op);
  }
  ~CustomFunction() { MPI_Op_free(&mpi_op); }
  MPI_Op& get_mpi_op() { return mpi_op; }
  MPI_Op mpi_op;
};

template <typename DataType, typename Op>
static inline std::vector<DataType>
allreduce(std::vector<DataType>& data, Op&& /*op*/,
          const MPIContext& ctx = MPIContext()) {
  TypeMapper<DataType> tm;

  const int is_commutative = 0;
  CustomFunction<is_commutative, Op, DataType> fun;
  std::vector<DataType> recv_data(data.size());
  {
    const std::size_t io_bytes = data.size() * sizeof(DataType);
    get_communication_tracker().add_volume(io_bytes, io_bytes);
  }
  MPI_Allreduce(data.data(), recv_data.data(), data.size(),
                tm.get_mpi_datatype(), fun.get_mpi_op(), ctx.communicator());
  return recv_data;
}

template <typename Op, typename DataType>
static inline std::vector<DataType>
allreduce(std::vector<DataType>& data, const MPIContext& ctx = MPIContext()) {
  return allreduce(data, Op{}, ctx); // TODO remove dependencies in own code
}

} // namespace hybridMST::mpi
