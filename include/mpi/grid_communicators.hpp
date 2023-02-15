#pragma once

#include <cassert>
#include <cmath>
#include <mpi.h>

#include "mpi/context.hpp"
#include <util/utils.hpp>

namespace hybridMST {
  // TODO rework all this to make it independent from power of two
namespace mpi {
///@brief Construct MPI communicator in a grid based on the given input
/// communicator.
/// Assume that the size (,i.e. p) of the communicator is a power of two p =
/// 2^k. The grid has 2^floor(k/2) columns and 2^ceil(k/2) rows. Each column and
/// each row is a separate MPI_Comm. Consecutive elements in the column
/// communicators are also consecutive in the input communicator.
/// Example: indices are the ranks of the input communicator
///   c0 c1 c2 c3
/// r0 0  4  8 12
/// r1 1  5  9 13
/// r2 2  6 10 14
/// r3 3  7 11 15
class PowerTwoGridCommunicators {
  friend PowerTwoGridCommunicators& get_power_two_grid_communicators();

public:
  [[nodiscard]] mpi::MPIContext get_row_ctx() const { return row_ctx; }
  [[nodiscard]] mpi::MPIContext get_col_ctx() const { return column_ctx; }
  [[nodiscard]] std::size_t num_columns() const { return row_ctx.size(); }
  [[nodiscard]] std::size_t num_rows() const { return column_ctx.size(); }
  [[nodiscard]] PEID column_id(int world_rank) const {
    return world_rank / num_pe_per_column;
  }
  [[nodiscard]] PEID row_id(int world_rank) const {
    return world_rank % num_pe_per_column;
  }

private:
  PowerTwoGridCommunicators(int rank, int size, MPI_Comm comm) {
    std::size_t log_size = std::log2(size);
    std::size_t num_columns = 1ull << (log_size / 2);
    num_pe_per_column = size / num_columns;
    [[maybe_unused]] const bool is_num_pe_power_of_two =
        (1ull << log_size) == static_cast<std::size_t>(size);

    assert(is_num_pe_power_of_two);

    // PEs in column-communicators must be consecutive
    int column_id = rank / num_pe_per_column;
    int row_id = rank % num_pe_per_column;
    MPI_Comm_split(comm, row_id, rank, &row_comm);
    MPI_Comm_split(comm, column_id, rank, &column_comm);
    row_ctx = mpi::MPIContext(row_comm);
    column_ctx = mpi::MPIContext(column_comm);
  }
  PowerTwoGridCommunicators(const PowerTwoGridCommunicators&) = delete;
  PowerTwoGridCommunicators(PowerTwoGridCommunicators&&) = delete;
  ~PowerTwoGridCommunicators() {
    // free_communicators();
    // TODO cannot call as this has to happen before MPI_Finalize
    // think of solution via global singleton or so ...
  }

  void free_communicators() {
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
  }
  std::size_t num_pe_per_column;
  MPI_Comm row_comm;
  MPI_Comm column_comm;
  mpi::MPIContext row_ctx;
  mpi::MPIContext column_ctx;
};

inline PowerTwoGridCommunicators& get_power_two_grid_communicators() {
  mpi::MPIContext ctx;
  static PowerTwoGridCommunicators grid_communicators(ctx.rank(), ctx.size(),
                                                      ctx.communicator());
  return grid_communicators;
}
} // namespace mpi
} // namespace hybridMST
