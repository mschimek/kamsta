#pragma once

#include <cmath>
#include <mpi.h>
#include <vector>

#include "mpi/allgather.hpp"
#include "mpi/broadcast.hpp"
#include "mpi/context.hpp"

namespace hybridMST {
namespace mpi {
///  Builds a 2D grid communicator with num_columns <= num_rows <= num_columns
///  + 1. The original ranks (world communicator) are column-wise consecutive,
///  e.g..:
///
///  for size = 7:
///     c0 c1 c2
///  r0 0  3  6
///  r1 1  4  7
///  r2 2
///
///  for size = 8:
///     c0 c1 c2
///  r0 0  3  6
///  r1 1  4  7
///  r2 2  5
///
///  for size = 9:
///     c0 c1 c2
///  r0 0  3  6
///  r1 1  4  7
///  r2 2  5  8
///
class TwoLevelColumnMajorCommunicator {
public:
  TwoLevelColumnMajorCommunicator(int rank, int size, MPI_Comm comm) {
    const double sqrt = std::sqrt(size);
    num_columns_ = std::floor(sqrt);
    num_rows_ = std::ceil(sqrt);
    if (size > num_columns_ * num_rows_) {
      ++num_columns_;
    }
    // assert(num_columns_ == num_rows_ || num_columns_ + 1 == num_rows_);
    // assert(size <= num_columns_ * num_rows_);

    const int remainder = size % num_columns_;
    num_large_columns_ = remainder == 0 ? num_columns_ : remainder;
    threshold_ = num_large_columns_ * num_rows_;
    row_idx_ = row_index(rank);
    col_idx_ = col_index(rank);

    MPI_Comm_split(comm, row_idx_, rank, &row_comm);
    MPI_Comm_split(comm, col_idx_, rank, &column_comm);
    row_ctx = mpi::MPIContext(row_comm);
    column_ctx = mpi::MPIContext(column_comm);
  }
  [[nodiscard]] int row_index() const { return row_idx_; }
  [[nodiscard]] int row_index(int rank) const {
    if (is_large_column(rank)) {
      return rank % num_rows_;
    }
    return (rank - threshold_) % (num_rows_ - 1);
  }
  [[nodiscard]] int col_index() const { return col_idx_; }
  [[nodiscard]] int col_index(int rank) const {
    if (is_large_column(rank)) {
      return rank / num_rows_;
    }
    return ((rank - threshold_) / (num_rows_ - 1)) + num_large_columns_;
  }
  [[nodiscard]] int row_size(int /*rank*/) const { return num_columns_; }
  [[nodiscard]] int col_size(int rank) const {
    if (is_large_column(rank)) {
      return num_rows_;
    } else {
      return num_rows_ - 1;
    }
  }
  [[nodiscard]] int large_col_size() const { return num_rows_; }
  /// colum ides are zero-indexed
  [[nodiscard]] int size_of_col_with(int col_id) const {
    if (col_id < num_large_columns_) {
      return num_rows_;
    } else {
      return num_rows_ - 1;
    }
  }
  [[nodiscard]] int num_cols() const { return num_columns_; }
  [[nodiscard]] int max_num_rows() const { return num_columns_; }

  /// colum ides are zero-indexed
  [[nodiscard]] int min_org_rank_of_column(int col_id) const {
    if (col_id < num_large_columns_) {
      return col_id * num_rows_;
    }

    return threshold_ + ((col_id - num_large_columns_) * (num_rows_ - 1));
  }
  [[nodiscard]] int min_org_rank_of_column() const {
    return min_org_rank_of_column(col_idx_);
  }
  [[nodiscard]] mpi::MPIContext get_row_ctx() const { return row_ctx; }
  [[nodiscard]] mpi::MPIContext get_col_ctx() const { return column_ctx; }
  void free_communicators() {
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&column_comm);
  }

private:
  bool is_large_column(int rank) const { return rank < threshold_; }
  int threshold_;
  int num_columns_;
  int num_large_columns_;
  int num_rows_;

  int row_idx_;
  int col_idx_;

  MPI_Comm row_comm;
  MPI_Comm column_comm;
  mpi::MPIContext row_ctx;
  mpi::MPIContext column_ctx;
};

inline TwoLevelColumnMajorCommunicator&
get_twolevel_columnmajor_communicators() {
  mpi::MPIContext ctx;
  static TwoLevelColumnMajorCommunicator twolevel_comm(ctx.rank(), ctx.size(),
                                                       ctx.communicator());
  return twolevel_comm;
}

/// Here we assume that send_data is the same on all PE within a column.
/// However we cannot simply perform an row-wise allgather as the last
/// row might be incomplete.
/// Instead a row-wise allgather on the first row (which is complete by
/// construction) is performed followed by a column-wise bcast.
template <typename DataType>
inline std::vector<DataType> row_wise_allgatherv_on_column_data(
    std::vector<DataType>& send_data,
    const TwoLevelColumnMajorCommunicator& comm) {
  const auto& column_ctx = comm.get_col_ctx();
  const auto& row_ctx = comm.get_row_ctx();
  if (column_ctx.rank() == 0) {
    send_data = mpi::allgatherv(send_data, row_ctx);
  } else {
    // only for consistency in communication volume tracking
    // as all PEs have to perform take part in a collective operation
    send_data = mpi::allgatherv(send_data, row_ctx);
    send_data.clear();
  }
  mpi::bcast(send_data, 0, column_ctx);
  return send_data;
}
/// Here we assume that send_data is the same on all PE within a column.
/// However we cannot simply perform an row-wise allgather as the last
/// row might be incomplete.
/// Instead a row-wise allgather on the first row (which is complete by
/// construction) is performed followed by a column-wise bcast.
template <typename DataType>
inline std::vector<DataType> row_wise_allgatherv_on_column_data(
    const DataType& send_data, const TwoLevelColumnMajorCommunicator& comm) {
  const auto& column_ctx = comm.get_col_ctx();
  const auto& row_ctx = comm.get_row_ctx();
  std::vector<DataType> buffer;
  if (column_ctx.rank() == 0) {
    buffer = mpi::allgather(send_data, row_ctx);
  } else {
    // only for consistency in communication volume tracking
    // as all PEs have to perform take part in a collective operation
    buffer = mpi::allgather(send_data, row_ctx);
    buffer.clear();
  }
  mpi::bcast(buffer, 0, column_ctx);
  return buffer;
}
} // namespace mpi
} // namespace hybridMST
