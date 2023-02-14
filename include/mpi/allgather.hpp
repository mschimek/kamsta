/*******************************************************************************
 * mpi/allgather.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <mpi.h>
#include <numeric>
#include <vector>

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include "util/communication_volume_measurements.hpp"

namespace hybridMST::mpi {

template <typename DataType>
inline std::vector<DataType> allgather(const DataType& send_data,
                                       const MPIContext& ctx = MPIContext()) {
  TypeMapper<DataType> dtm;
  std::vector<DataType> receive_data(ctx.size());
  {
    const std::size_t recv_bytes = sizeof(DataType) * ctx.size();
    const std::size_t sent_bytes = sizeof(DataType);
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }
  MPI_Allgather(&send_data, 1, dtm.get_mpi_datatype(), receive_data.data(), 1,
                dtm.get_mpi_datatype(), ctx.communicator());
  return receive_data;
}

template <typename DataType>
static inline std::vector<DataType>
allgatherv(std::vector<DataType>& send_data,
           const MPIContext& ctx = MPIContext()) {

  int32_t local_size = send_data.size();
  std::vector<int32_t> receiving_sizes = allgather(local_size, ctx);

  std::vector<int32_t> receiving_offsets(ctx.size(), 0);
  for (size_t i = 1; i < receiving_sizes.size(); ++i) {
    receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
  }

  std::vector<DataType> receiving_data(receiving_sizes.back() +
                                       receiving_offsets.back());

  {
    const std::size_t recv_bytes =
        sizeof(DataType) *
        std::accumulate(receiving_sizes.begin(), receiving_sizes.end(), 0ull);
    const std::size_t sent_bytes = sizeof(DataType) * local_size;
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }
  TypeMapper<DataType> dtm;
  MPI_Allgatherv(send_data.data(), local_size, dtm.get_mpi_datatype(),
                 receiving_data.data(), receiving_sizes.data(),
                 receiving_offsets.data(), dtm.get_mpi_datatype(),
                 ctx.communicator());

  return receiving_data;
}

template <typename Container,
          typename DataType = typename Container::value_type>
static inline Container allgatherv(Container& send_data,
                                   const MPIContext& ctx = MPIContext()) {

  int32_t local_size = send_data.size();
  std::vector<int32_t> receiving_sizes = allgather(local_size, ctx);

  std::vector<int32_t> receiving_offsets(ctx.size(), 0);
  for (size_t i = 1; i < receiving_sizes.size(); ++i) {
    receiving_offsets[i] = receiving_offsets[i - 1] + receiving_sizes[i - 1];
  }

  Container receiving_data(receiving_sizes.back() + receiving_offsets.back());

  {
    const std::size_t recv_bytes =
        sizeof(DataType) *
        std::accumulate(receiving_sizes.begin(), receiving_sizes.end(), 0ull);
    const std::size_t sent_bytes = sizeof(DataType) * local_size;
    get_communication_tracker().add_volume(sent_bytes, recv_bytes);
  }

  TypeMapper<DataType> dtm;
  MPI_Allgatherv(send_data.data(), local_size, dtm.get_mpi_datatype(),
                 receiving_data.data(), receiving_sizes.data(),
                 receiving_offsets.data(), dtm.get_mpi_datatype(),
                 ctx.communicator());

  return receiving_data;
}
} // namespace hybridMST::mpi
