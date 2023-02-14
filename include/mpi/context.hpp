#pragma once
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include "mpi.h"

namespace hybridMST::mpi {

inline int& get_nb_threads() {
  static int nb_threads = 1;
  return nb_threads;
}

constexpr int ROOT = 0;
class MPIContext {
public:
  MPIContext() : MPIContext(MPI_COMM_WORLD) {}
  MPIContext(MPI_Comm comm) : communicator_{comm} {
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
      int provided_thread_support = -1;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED,
                      &provided_thread_support);
      if (provided_thread_support < MPI_THREAD_FUNNELED) {
        std::cout << "thread support level is: " << provided_thread_support
                  << " we need: " << MPI_THREAD_FUNNELED << " --> abort."
                  << std::endl;
        std::abort();
      }
    }
    MPI_Comm_rank(communicator_, &rank_);
    MPI_Comm_size(communicator_, &size_);
  }
  int rank() const { return rank_; }

  int size() const { return size_; }

  void set_threads_per_mpi_process(int threads_per_mpi_process) {
    auto& tmp = get_nb_threads();
    tmp = threads_per_mpi_process;
  }
  int threads_per_mpi_process() const { return get_nb_threads(); }
  int total_size() const { return size() * threads_per_mpi_process(); }

  bool is_root(int root_rank = ROOT) const { return rank_ == root_rank; }

  MPI_Comm communicator() const { return communicator_; }
  static bool is_initialized() {
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    return is_mpi_initialized;
  }

  static void initialize() {
    if (!is_initialized())
      MPI_Init(nullptr, nullptr);
  }

  static bool is_finalized() {
    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);
    return is_mpi_finalized;
  }

  static constexpr int max_mpi_int() { return std::numeric_limits<int>::max(); }

  static void finalize() {
    if (!is_finalized())
      MPI_Finalize();
  }

  void barrier() const { MPI_Barrier(communicator_); }

  void abort(const std::string& msg) const {
    if (is_root())
      std::cout << msg << " -> abort" << std::endl;
    MPI_Abort(communicator_, EXIT_FAILURE);
  }

  void mpi_assert(bool condition,
                  const std::string& msg = "assertion failed") const {
    if (condition)
      return;
    std::cout << "On rank: " << rank() << " \n\t" << msg << "\n\b -> abort"
              << std::endl;
    MPI_Abort(communicator_, EXIT_FAILURE);
  }
  void mpi_assert(bool condition, const std::string& location,
                  const std::string& msg) const {
    if (condition)
      return;
    std::cout << "On rank: " << rank() << " at: " << location << "\n\t" << msg
              << "\n\b -> abort" << std::endl;
    MPI_Abort(communicator_, EXIT_FAILURE);
  }

  MPIContext split_communicator(int color) {
    MPI_Comm new_comm;
    MPI_Comm_split(communicator(), color, rank(), &new_comm);
    return MPIContext(new_comm);
  }

  template <typename Function>
  void execute_in_order(Function&& func, bool with_sleep = true,
                        bool with_rank_info = true) const {
    for (int i = 0; i < size(); ++i) {
      if (with_sleep)
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
      MPI_Barrier(communicator());
      if (i == rank()) {
        if (with_rank_info)
          std::cout << "On rank " << rank() << std::endl;
        func();
      }
    }
  }

private:
  int rank_ = -1;
  int size_ = 0;
  MPI_Comm communicator_;
};

} // namespace hybridMST::mpi
