#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

#include <mpi.h>

#include "mpi/context.hpp"

namespace hybridMST::mpi {
template <typename Payload_> struct Message {
  using Payload = Payload_;
  Message() {}

  Message(Payload_ payload_) : sender_receiver{0u}, payload{payload_} {}

  Message(std::uint32_t sender_receiver_, Payload_ payload_)
      : sender_receiver{sender_receiver_}, payload{payload_} {}

  void set_sender(std::uint32_t sender) {
    sender = sender << offset;
    sender_receiver |= sender;
  }

  void set_receiver(std::uint32_t receiver) { sender_receiver |= receiver; }

  [[nodiscard]] std::uint32_t get_sender() const {
    return sender_receiver >> offset;
  }

  [[nodiscard]] std::uint32_t get_receiver() const {
    constexpr std::uint32_t ones = ~std::uint32_t{0};
    constexpr std::uint32_t lower_half = ones >> offset;
    return sender_receiver & lower_half;
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const Message<Payload_>& msg) {
    return out << "(" << msg.get_sender() << ", " << msg.get_receiver() << ": "
               << msg.payload << ")";
  }

  std::uint32_t sender_receiver; // [sender,receiver]
  Payload payload;
  static constexpr std::uint32_t offset = 16u;
};

class TwoLevelCommunicator {
public:
  TwoLevelCommunicator(int rank, int size, MPI_Comm comm) {
    const double sqrt = std::sqrt(size);
    floor_sqrt = std::floor(sqrt);
    const bool is_square = static_cast<int>(floor_sqrt * floor_sqrt) == size;
    int row_num = rank / floor_sqrt;
    int column_num = rank % floor_sqrt;
    if (!is_square && rank >= (floor_sqrt * floor_sqrt)) {
      row_num = rank % floor_sqrt; // virtual group
    }
    MPI_Comm_split(comm, row_num, rank, &row_comm);
    MPI_Comm_split(comm, column_num, rank, &column_comm);
    row_ctx = MPIContext(row_comm);
    column_ctx = MPIContext(column_comm);
  }
  [[nodiscard]] int row_index(int destination_rank) const {
    return destination_rank / floor_sqrt;
  }
  [[nodiscard]] int col_index(int destination_rank) const {
    return destination_rank % floor_sqrt;
  }
  [[nodiscard]] MPIContext get_row_ctx() const { return row_ctx; }
  [[nodiscard]] MPIContext get_col_ctx() const { return column_ctx; }
  void free_communicators() {
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
  }

private:
  std::size_t floor_sqrt;
  MPI_Comm row_comm;
  MPI_Comm column_comm;
  MPIContext row_ctx;
  MPIContext column_ctx;
};

inline TwoLevelCommunicator& get_grid_communicators() {
  mpi::MPIContext ctx;
  static TwoLevelCommunicator two_level_comm(ctx.rank(), ctx.size(),
                                             ctx.communicator());
  return two_level_comm;
}
} // namespace hybridMST::mpi
