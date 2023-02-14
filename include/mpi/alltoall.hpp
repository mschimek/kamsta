#pragma once
#include <omp.h>
#include <tbb/concurrent_vector.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "definitions.hpp"
#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include "util/allocators.hpp"
#include "util/communication_volume_measurements.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

namespace hybridMST {
namespace mpi {
// from https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html

template <typename T> struct PreallocMessages {
  using MessageType = T;
  PreallocMessages(const std::vector<uint64_t>& send_counts)
      : send_counts{send_counts},
        buffer(std::accumulate(send_counts.begin(), send_counts.end(), 0ull)) {}
  PreallocMessages(const std::vector<uint64_t>& send_counts, uint64_t count_sum)
      : send_counts{send_counts}, buffer(count_sum) {}
  T& operator[](const std::size_t idx) { return buffer[idx]; }
  const T& operator[](const std::size_t idx) const { return buffer[idx]; }
  std::vector<uint64_t> send_counts;
  std::vector<T, default_init_allocator<T>> buffer;
};

template <typename T> struct PreallocSparseMessages {
  using PeCount = std::pair<int, uint64_t>;
  PreallocSparseMessages(const std::vector<PeCount>& send_counts)
      : send_counts{send_counts},
        buffer(
            std::accumulate(send_counts.begin(), send_counts.end(), 0ull,
                            [](const uint64_t& accu, const PeCount& pe_count) {
                              return accu + pe_count.second;
                            })) {}
  T& operator[](const std::size_t idx) { return buffer[idx]; }
  const T& operator[](const std::size_t idx) const { return buffer[idx]; }
  std::vector<PeCount> send_counts;
  std::vector<T> buffer;
};

template <typename T> struct RecvMessage {
  T* data;
  uint64_t count;
  T* begin() { return data; }
  T* end() { return data + count; }
  const T* begin() const { return begin(); }
  const T* end() const { return end(); }
};

template <typename T> struct RecvMessages {
  using value_type = T;
  RecvMessages() = default;
  RecvMessages(const std::vector<int32_t>& recv_displacements,
               const int32_t recv_counts_sum)
      : recv_displacements{recv_displacements}, buffer(recv_counts_sum) {}
  RecvMessages(const std::vector<non_init_vector<T>>& recv_data)
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

template <typename DataType>
inline std::vector<DataType> alltoall(const std::vector<DataType>& send_data,
                                      const MPIContext& ctx = MPIContext()) {
  std::vector<DataType> receive_data(send_data.size(), 0);
  std::size_t sum_elements_sent =
      std::accumulate(send_data.begin(), send_data.end(), 0ull);
  std::size_t sum_io_bytes = sum_elements_sent * sizeof(DataType);

  get_communication_tracker().add_volume(sum_io_bytes, sum_io_bytes);
  TypeMapper<DataType> tm;
  MPI_Alltoall(send_data.data(), send_data.size() / ctx.size(),
               tm.get_mpi_datatype(), receive_data.data(),
               send_data.size() / ctx.size(), tm.get_mpi_datatype(),
               ctx.communicator());
  return receive_data;
}

template <typename DataType>
inline RecvMessages<DataType>
alltoallv(const PreallocMessages<DataType>& send_messages,
          const mpi::MPIContext& ctx = mpi::MPIContext()) {
  // get_timer().start_phase_measurement("alltoallv");
  std::vector<int32_t> send_counts(send_messages.send_counts.size());
  for (size_t i = 0; i < send_counts.size(); ++i) {
    send_counts[i] = static_cast<int>(send_messages.send_counts[i]);
  }
  const auto& send_buffer = send_messages.buffer;
  std::vector<int32_t> receive_counts = alltoall(send_counts, ctx);
  std::vector<int32_t> send_displacements(send_counts.size(), 0);
  std::vector<int32_t> receive_displacements(send_counts.size(), 0);
  for (size_t i = 1; i < send_counts.size(); ++i) {
    send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
    receive_displacements[i] =
        receive_displacements[i - 1] + receive_counts[i - 1];
  }
  const uint64_t recv_counts_sum =
      receive_counts.back() + receive_displacements.back();

  RecvMessages<DataType> recv_messages(receive_displacements, recv_counts_sum);
  get_communication_tracker().add_volume_print(send_counts, receive_counts,
                                               sizeof(DataType));
  TypeMapper<DataType> dtm;
  MPI_Alltoallv(send_buffer.data(), send_counts.data(),
                send_displacements.data(), dtm.get_mpi_datatype(),
                recv_messages.buffer.data(), receive_counts.data(),
                receive_displacements.data(), dtm.get_mpi_datatype(),
                ctx.communicator());

  // get_timer().stop_phase_measurement("alltoallv");
  return recv_messages;
}
template <typename DataType>
RecvMessages<DataType>
sparse_alltoall(const PreallocMessages<DataType>& send_messages, int tag,
                mpi::MPIContext ctx = mpi::MPIContext{}) {
  TypeMapper<DataType> tm;

  get_timer().start_phase_measurement("alltoallv");
  std::vector<int32_t> send_counts(send_messages.send_counts.size());
  for (size_t i = 0; i < send_counts.size(); ++i) {
    const std::size_t send_count = send_messages.send_counts[i];
    MPI_ASSERT_(send_count < std::numeric_limits<int32_t>::max(), send_count);
    send_counts[i] = static_cast<int32_t>(send_messages.send_counts[i]);
  }
  const auto& send_buffer = send_messages.buffer;
  std::vector<int32_t> send_displacements(send_counts.size(), 0);
  for (size_t i = 1; i < send_counts.size(); ++i) {
    send_displacements[i] = send_displacements[i - 1] + send_counts[i - 1];
  }

  std::vector<MPI_Request> requests;
  requests.reserve(ctx.size());

  for (int i = 0; i < ctx.size(); i++) {
    if (send_counts[i] == 0)
      continue;
    requests.emplace_back(MPI_Request{});
    MPI_Issend(send_buffer.data() + send_displacements[i],
               static_cast<int>(send_counts[i]), tm.get_mpi_datatype(), i, tag,
               ctx.communicator(), &requests.back());
  }
  {
    auto send_counts_all = mpi::allgatherv(send_counts);
    if (ctx.rank() == 0) {
      for (int i = 0; i < ctx.size(); ++i) {
        std::cout << "rank: " << i << " ";
        for (int j = 0; j < ctx.size(); ++j) {
          std::cout << "(" << j << ", " << send_counts_all[i * ctx.size() + j]
                    << "), ";
        }
        std::cout << std::endl;
      }
    }
  }
  std::vector<non_init_vector<DataType>> recv_buffers(ctx.size());
  const std::size_t nb_messages = requests.size();
  std::vector<MPI_Status> statuses(nb_messages);
  int isend_done = 0;
  while (isend_done == 0) {
    // Check for messages
    int iprobe_success = 1;
    while (iprobe_success > 0) {
      iprobe_success = 0;
      MPI_Status status{};
      MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
      if (iprobe_success > 0) {
        int msg_count;
        MPI_Get_count(&status, tm.get_mpi_datatype(), &msg_count);
        non_init_vector<DataType> message(msg_count);
        MPI_Status rst{};
        MPI_Recv(message.data(), msg_count, tm.get_mpi_datatype(),
                 status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &rst);
        recv_buffers[status.MPI_SOURCE] = std::move(message);
      }
    }
    // Check if all ISend successful
    isend_done = 0;
    MPI_Testall(nb_messages, requests.data(), &isend_done, MPI_STATUSES_IGNORE);
  }
  ctx.barrier();
  if (ctx.rank() == 0) {
    std::cout << "sents posted" << std::endl;
  }
  MPI_Request barrier_request;
  MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);

  int ibarrier_done = 0;
  while (ibarrier_done == 0) {
    int iprobe_success = 1;
    while (iprobe_success > 0) {
      iprobe_success = 0;
      MPI_Status status{};
      MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &iprobe_success, &status);
      if (iprobe_success > 0) {
        int msg_count;
        MPI_Get_count(&status, tm.get_mpi_datatype(), &msg_count);
        non_init_vector<DataType> message(msg_count);
        MPI_Recv(message.data(), msg_count, tm.get_mpi_datatype(),
                 status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);

        recv_buffers[status.MPI_SOURCE] = std::move(message);
      }
    }
    // Check if all reached Ibarrier
    MPI_Status tst{};
    MPI_Test(&barrier_request, &ibarrier_done, &tst);
    MPI_ASSERT(ctx, tst.MPI_ERROR == MPI_SUCCESS, "mpi_test barrier failed");
  }
  get_timer().stop_phase_measurement("alltoallv");
  return RecvMessages<DataType>(recv_buffers);
}

template <typename T, typename Filter, typename Transformer, typename Pe,
          typename TransformedT = T>
inline RecvMessages<T>
twopass_alltoallv(const std::vector<T>& data, Filter&& filter,
                  Transformer&& transformer, Pe&& pe_getter) {
  MPIContext ctx;
  get_timer().start_phase_measurement("twopass_loop1");
  std::vector<uint64_t> send_counts(ctx.size(), 0);
  for (std::size_t i = 0; i < data.size(); ++i) {
    const auto& elem = data[i];
    if (filter(elem, i))
      continue;
    ++send_counts[pe_getter(elem, i)];
  }
  get_timer().stop_phase_measurement("twopass_loop1");
  get_timer().start_phase_measurement("twopass_accounting");
  PreallocMessages<TransformedT> send_messages(send_counts);
  std::vector<uint64_t> offsets_send_counts(ctx.size(), 0);
  std::exclusive_scan(send_counts.begin(), send_counts.end(),
                      offsets_send_counts.begin(), 0ull);
  get_timer().stop_phase_measurement("twopass_accounting");
  get_timer().start_phase_measurement("twopass_loop2");
  for (std::size_t i = 0; i < data.size(); ++i) {
    const auto& elem = data[i];
    if (filter(elem, i))
      continue;
    const auto& pe = pe_getter(elem, i);
    send_messages[offsets_send_counts[pe]] = transformer(elem);
    ++offsets_send_counts[pe];
  }
  get_timer().stop_phase_measurement("twopass_loop2");
  return alltoallv(send_messages);
}

template <typename T>
bool check(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].size() != rhs[i].size()) {
      std::cout << i << " lhs: " << lhs[i].size() << std::endl;
      std::cout << i << " rhs: " << rhs[i].size() << std::endl;
    }
    if (lhs[i] != rhs[i]) {
      for (const auto& elem : lhs[i])
        std::cout << elem << ", ";
      std::cout << std::endl;
      for (const auto& elem : rhs[i])
        std::cout << elem << ", ";
      std::cout << std::endl;
      return false;
    }
  }
  return true;
}

struct Task {
  Task() = default;
  Task(std::size_t task_id_, std::size_t num_elements, std::size_t num_tasks,
       std::size_t num_destinations)
      : task_id{task_id_}, idx_begin{task_id * (num_elements / num_tasks)},
        idx_end{((task_id + 1) * (num_elements / num_tasks)) +
                ((task_id_ + 1 == num_tasks) ? (num_elements % num_tasks) : 0)},
        send_counts(num_destinations, 0ull) {}
  std::size_t task_id;
  std::size_t idx_begin;
  std::size_t idx_end;
  std::vector<std::size_t> send_counts;
  friend std::ostream& operator<<(std::ostream& out, const Task& task) {
    return out << "(" << task.task_id << ", " << task.idx_begin << " "
               << task.idx_end << " " << task.send_counts.size() << ")";
  }
};

std::vector<Task, tbb::cache_aligned_allocator<Task>> inline create_tasks(
    std::size_t num_elements, std::size_t num_tasks,
    std::size_t num_destinations) {
  mpi::MPIContext ctx;
  using CacheAlignedVector =
      std::vector<Task, tbb::cache_aligned_allocator<Task>>;
  CacheAlignedVector tasks(num_tasks);
  // SEQ_EX(ctx, PRINT_VAR(num_tasks); PRINT_VAR(num_elements);
  // PRINT_CONTAINER_WITH_INDEX(tasks););
#pragma omp parallel for
  for (std::size_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task(i, num_elements, num_tasks, num_destinations);
  };
  // SEQ_EX(ctx, PRINT_VAR(num_tasks); PRINT_VAR(num_elements);
  // PRINT_CONTAINER_WITH_INDEX(tasks););
  return tasks;
}

inline void analysis(std::vector<size_t> send_counts) {
  std::sort(send_counts.begin(), send_counts.end());
  const auto sum_counts =
      std::accumulate(send_counts.begin(), send_counts.end(), 0ull);
  {
    const auto& count = get_timer().get_phase_add_count();
    get_timer().add_phase("send_count_min", count, send_counts.front(),
                          {Timer::DatapointsOperation::ID});
  }
  {
    const auto& count = get_timer().get_phase_add_count();
    get_timer().add_phase("send_count_max", count, send_counts.back(),
                          {Timer::DatapointsOperation::ID});
  }
  {
    const auto& count = get_timer().get_phase_add_count();
    get_timer().add_phase("send_count_median", count,
                          send_counts[send_counts.size() / 2],
                          {Timer::DatapointsOperation::ID});
  }
  {
    const auto& count = get_timer().get_phase_add_count();
    get_timer().add_phase("send_count_sum", count, sum_counts,
                          {Timer::DatapointsOperation::ID});
  }
}

template <
    typename Container, typename Filter, typename Transformer, typename DstCalculator>
inline auto twopass_alltoallv_openmp_special(
    Container&& data, Filter&& filter, Transformer&& transformer,
    DstCalculator&& dstCalculator, mpi::MPIContext ctx = mpi::MPIContext{},
    const bool use_dense = true, int tag = 42) {
  using Container_ = std::decay_t<Container>;
  using TransformedT =
      std::invoke_result_t<Transformer, const typename Container_::value_type&,
                           const std::size_t&>;
          std::size_t nb_threads = ctx.threads_per_mpi_process();
  std::size_t max_nb_dst = ctx.size();
  auto tasks = create_tasks(data.size(), nb_threads, max_nb_dst);
  // get_timer().start_phase_measurement("twopass_loop1");

  non_init_vector<PEID> destinations(data.size());
#pragma omp parallel for
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    // for (std::size_t i = r.begin(); i < r.end(); ++i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      if (filter(elem, j)) {
        destinations[j] = -1;
        continue;
      }
      const PEID destination_pe = dstCalculator(elem, j);
      destinations[j] = destination_pe;
      ++task.send_counts[destination_pe];
    }
  }

  // get_timer().stop_phase_measurement("twopass_loop1");
  // get_timer().start_phase_measurement("twopass_accounting");

  // compute 2D-exclusive scan cache-friendly
  std::vector<size_t> init(max_nb_dst, 0ull);
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    for (std::size_t j = 0; j < max_nb_dst; ++j) {
      const auto v = init[j];
      init[j] += tasks[i].send_counts[j];
      tasks[i].send_counts[j] = v;
    }
  }
  // analysis(init);
  //  init[j] now contains the global nb elements to be sent to destination j
  PreallocMessages<TransformedT> send_messages(init);
  // get_timer().stop_phase_measurement("twopass_accounting");
  // get_timer().start_phase_measurement("twopass_loop2");
  std::vector<std::size_t> send_displs(max_nb_dst, 0);
  std::exclusive_scan(init.begin(), init.end(), send_displs.begin(), 0ull);

#pragma omp parallel for
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    auto& task = tasks[i];
    for (std::size_t j = task.idx_begin; j < task.idx_end; ++j) {
      const auto& elem = data[j];
      const auto dst = destinations[j];
      if (dst == -1) {
        continue;
      }
      const auto idx = send_displs[dst] + task.send_counts[dst];
      send_messages[idx] = transformer(elem, j);
      ++task.send_counts[dst];
    }
  }
  // get_timer().stop_phase_measurement("twopass_loop2");
  if constexpr (std::is_rvalue_reference_v<Container&&>) {
    dump(data);
  }
  if (use_dense) {
    return alltoallv(send_messages, ctx);
  } else {
    return sparse_alltoall(send_messages, tag);
  }
}

} // namespace mpi
} // namespace hybridMST
