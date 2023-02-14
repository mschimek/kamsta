#pragma once

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include "util/communication_volume_measurements.hpp"

namespace hybridMST::mpi {
template <typename DataType>
inline DataType send_recv(const int send_to, const int recv_from,
                          const DataType& send_elem,
                          const MPIContext& ctx = MPIContext()) {
  TypeMapper<DataType> dtm;
  DataType recv_elem;
  int send_tag = 0;
  int recv_tag = 0;

  {
    get_communication_tracker().add_volume(sizeof(DataType), sizeof(DataType));
  }
  MPI_Sendrecv(&send_elem, 1, dtm.get_mpi_datatype(), send_to, send_tag,
               &recv_elem, 1, dtm.get_mpi_datatype(), recv_from, recv_tag,
               ctx.communicator(), MPI_STATUS_IGNORE);
  return recv_elem;
}

template <typename Container>
inline Container send_recv_v(const int send_to, const int recv_from,
                             const Container& send_elems,
                             const MPIContext& ctx = MPIContext()) {
  using DataType = typename Container::value_type;
  TypeMapper<DataType> dtm;
  int send_tag = 0;
  int recv_tag = 0;

  const int send_size = send_elems.size();
  const int recv_size = send_recv(send_to, recv_from, send_size, ctx);
  Container recv_elems(recv_size);
  {
    get_communication_tracker().add_volume(send_size * sizeof(DataType),
                                           recv_size * sizeof(DataType));
  }
  MPI_Sendrecv(send_elems.data(), send_size, dtm.get_mpi_datatype(), send_to,
               send_tag, recv_elems.data(), recv_size, dtm.get_mpi_datatype(),
               recv_from, recv_tag, ctx.communicator(), MPI_STATUS_IGNORE);
  return recv_elems;
}
} // namespace hybridMST::mpi
