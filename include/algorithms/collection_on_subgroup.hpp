#include <algorithm>
#include <mpi/context.hpp>
#include <mpi/type_handling.hpp>

#include "definitions.hpp"

namespace hybridMST {
template <typename T> struct GatherMessages {
  GatherMessages() = default;
  GatherMessages(const std::vector<int32_t>& recv_displacements,
                 const int32_t recv_counts_sum)
      : recv_displacements{recv_displacements}, buffer(recv_counts_sum) {}
  GatherMessages(const std::vector<non_init_vector<T>>& recv_data)
      : recv_displacements(recv_data.size()) {
    for (int i = 1; i < recv_data.size(); ++i) {
      recv_displacements[i] =
          recv_displacements[i - 1] + recv_data[i - 1].size();
    }
    non_init_vector<T> buffer_tmp(recv_data.back().size() +
                                  recv_displacements.back());
    for (int i = 0; i < recv_data.size(); ++i) {
      std::copy_n(recv_data[i].begin(), recv_data[i].size(),
                  buffer_tmp.begin() + recv_displacements[i]);
    }
    buffer = std::move(buffer_tmp);
  }
  [[nodiscard]] int get_pe(uint64_t idx) const noexcept {
    const auto it = std::upper_bound(recv_displacements.begin(),
                                     recv_displacements.end(), idx);
    return std::distance(recv_displacements.begin(), it) - 1;
  }
  std::vector<int32_t> recv_displacements;
  non_init_vector<T> buffer;
};

template <typename Container, typename DataType = typename Container::value_type>
GatherMessages<DataType>
gather_on_subgroup(Container& data, const mpi::MPIContext& ctx,
                   std::size_t reduction_factor) {
  mpi::TypeMapper<DataType> tm;
  int tag = 111;
  std::size_t receiver_range = ctx.size() / reduction_factor;
  const bool is_receiver = ctx.rank() < receiver_range;
  const PEID comm_partner =
      ctx.rank() % receiver_range; // has to be power of two
  if(!is_receiver) {
  MPI_Ssend(data.data(), static_cast<int>(data.size()), tm.get_mpi_datatype(),
             comm_partner, tag, ctx.communicator());
  return GatherMessages<DataType>{};
  }
  std::vector<non_init_vector<DataType>> recv_data(reduction_factor - 1);
  int num_recv_msg = 0;
  while(num_recv_msg < reduction_factor - 1) {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, tag, ctx.communicator(), &status);
    int msg_size = 0;
    MPI_Get_count(&status, tm.get_mpi_datatype(), &msg_size);
    non_init_vector<DataType> message(msg_size);
    MPI_Recv(message.data(), msg_size, tm.get_mpi_datatype(),
                 status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    recv_data[num_recv_msg++] = std::move(message);
  }
  return GatherMessages(recv_data);
}

} // namespace hybridMST
