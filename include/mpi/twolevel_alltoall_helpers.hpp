#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>

#include <mpi.h>

#include "mpi/context.hpp"

namespace hybridMST::mpi {
template <typename Payload_> class Message {
public:
  using Payload = Payload_;
  Message() {}

  Message(Payload_ payload) : sender_{0u}, receiver_{0u}, payload_{payload} {}

  Message(std::uint32_t sender, std::uint32_t receiver, Payload_ payload)
      : sender_{sender}, receiver_{receiver}, payload_{payload} {}

  void set_sender(std::uint32_t sender) { sender_ = sender; }

  void set_receiver(std::uint32_t receiver) { receiver_ = receiver; }

  [[nodiscard]] std::uint32_t get_sender() const { return sender_; }

  [[nodiscard]] std::uint32_t get_receiver() const { return receiver_; }
  void swap_sender_and_receiver() { std::swap(sender_, receiver_); }

  const Payload& payload() const { return payload_; }

  friend std::ostream& operator<<(std::ostream& out,
                                  const Message<Payload_>& msg) {
    return out << "(" << msg.get_sender() << ", " << msg.get_receiver() << ": "
               << msg.payload << ")";
  }

private:
  std::uint32_t sender_;
  std::uint32_t receiver_;
  Payload payload_;
  static constexpr std::uint32_t offset = 16u;
};

class TwoLevelCommunicator {
public:
  TwoLevelCommunicator(int rank, int size, MPI_Comm comm) {
    const double sqrt = std::sqrt(size);
    const std::size_t floor_sqrt = std::floor(sqrt);
    const std::size_t ceil_sqrt = std::ceil(sqrt);
    // if size exceeds the threshold we can afford one more column
    const std::size_t threshold = floor_sqrt * ceil_sqrt;
    number_columns_ =
        (static_cast<std::size_t>(size) < threshold) ? floor_sqrt : ceil_sqrt;
    const std::size_t num_pe_in_small_column = size / number_columns_;
    int row_num = rank / number_columns_;
    int column_num = rank % number_columns_;
    if (static_cast<std::size_t>(rank) >=
        (number_columns_ * num_pe_in_small_column)) {
      row_num = rank % number_columns_; // virtual group
    }
    MPI_Comm_split(comm, row_num, rank, &row_comm);
    MPI_Comm_split(comm, column_num, rank, &column_comm);
    row_ctx = MPIContext(row_comm);
    column_ctx = MPIContext(column_comm);
  }
  [[nodiscard]] int row_index(int destination_rank) const {
    return destination_rank / number_columns_;
  }
  [[nodiscard]] int col_index(int destination_rank) const {
    return destination_rank % number_columns_;
  }
  [[nodiscard]] MPIContext get_row_ctx() const { return row_ctx; }
  [[nodiscard]] MPIContext get_col_ctx() const { return column_ctx; }
  void free_communicators() {
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
  }

private:
  std::size_t number_columns_;
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
