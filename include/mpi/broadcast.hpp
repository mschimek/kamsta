#pragma once

#include <vector>

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include <util/communication_volume_measurements.hpp>

namespace hybridMST::mpi {

template <typename DataType>
inline DataType bcast(const DataType& data, int root,
                      const MPIContext& ctx = MPIContext()) {
  DataType elem = data;
  TypeMapper<DataType> dtm;
  get_communication_tracker().add_volume(
      ctx.rank() == root ? sizeof(DataType) : 0, sizeof(DataType));
  MPI_Bcast(&elem, 1, dtm.get_mpi_datatype(), root, ctx.communicator());
  return elem;
}

template <typename DataType, typename Allocator>
inline void bcast(std::vector<DataType, Allocator>& data, int root,
                  const MPIContext& ctx = MPIContext()) {
  int num_elem = data.size();
  num_elem = bcast(num_elem, root, ctx);
  if (ctx.rank() != root)
    data.resize(num_elem);
  TypeMapper<DataType> dtm;
  get_communication_tracker().add_volume(
      ctx.rank() == root ? num_elem * sizeof(DataType) : 0, num_elem * sizeof(DataType));

  MPI_Bcast(data.data(), num_elem, dtm.get_mpi_datatype(), root,
            ctx.communicator());
}
} // namespace hybridMST::mpi
