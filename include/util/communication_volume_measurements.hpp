#pragma once

#include <array>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "mpi/context.hpp"
#include "mpi/type_handling.hpp"
#include "util/macros.hpp"

namespace hybridMST {
class CommunicationVolumeTracker {
  struct SentRecvId {
    std::size_t num_sent_bytes = 0;
    std::size_t num_recv_bytes = 0;
    std::array<char, 30> id;

    friend std::ostream& operator<<(std::ostream& out, const SentRecvId& elem) {
      std::string id = ""; //(elem.id.begin(), elem.id.end());
      return out << "sent bytes: " << elem.num_sent_bytes
                 << " recv bytes: " << elem.num_recv_bytes << " id: " << id;
    }
  };

  struct AnalysisResult {
    std::size_t max_sent_bytes = 0;
    std::size_t max_recv_bytes = 0;
    std::size_t max_io_bytes = 0;
    std::size_t sum_sent_bytes = 0;
    std::size_t sum_recv_bytes = 0;
    friend std::ostream& operator<<(std::ostream& out,
                                    const AnalysisResult& elem) {
      out << "max sent bytes: " << elem.max_sent_bytes << " ";
      out << "max recv bytes: " << elem.max_recv_bytes << " ";
      out << "max io   bytes: " << elem.max_io_bytes << " ";
      out << "sum sent bytes: " << elem.sum_sent_bytes << " ";
      out << "sum recv bytes: " << elem.sum_sent_bytes << " ";
      return out;
    }
  };
  struct MetaAnalysisResult {
    std::size_t max_sent_bytes = 0;
    std::size_t max_recv_bytes = 0;
    std::size_t max_io_bytes = 0;
    std::size_t sum_sent_bytes = 0;
    std::size_t sum_recv_bytes = 0;
    std::size_t sum_max_sent_bytes = 0;
    std::size_t sum_max_recv_bytes = 0;
    std::size_t sum_max_io_bytes = 0;
  };

public:
  template <typename SendCounts, typename RecvCounts>
  void add_volume(const SendCounts& send_counts, const RecvCounts& recv_counts,
                  const std::size_t num_bytes_object) {
    const std::size_t num_sent_bytes =
        std::accumulate(send_counts.begin(), send_counts.end(), 0ull);
    const std::size_t num_recv_bytes =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0ull);
    add_volume(num_sent_bytes * num_bytes_object,
               num_recv_bytes * num_bytes_object);
  }
  template <typename SendCounts, typename RecvCounts>
  void add_volume_print(const SendCounts& send_counts,
                        const RecvCounts& recv_counts,
                        const std::size_t num_bytes_object) {
    const std::size_t num_sent_bytes =
        std::accumulate(send_counts.begin(), send_counts.end(), 0ull);
    const std::size_t num_recv_bytes =
        std::accumulate(recv_counts.begin(), recv_counts.end(), 0ull);
    mpi::MPIContext ctx;
    // SEQ_EX(ctx,
    // std::cout << ctx.rank() << "send: " << (num_sent_bytes *
    // num_bytes_object)
    //           << " recv: " << (num_recv_bytes * num_bytes_object) <<
    //           std::endl;);
    add_volume(num_sent_bytes * num_bytes_object,
               num_recv_bytes * num_bytes_object);
  }
  void add_volume(std::size_t num_sent_bytes, std::size_t num_recv_bytes) {
    SentRecvId elem;
    elem.num_sent_bytes = num_sent_bytes;
    elem.num_recv_bytes = num_recv_bytes;
    id_measurment[std::to_string(tracking_number++)] =
        elem; // could be agumented with phases
  }
  std::uint64_t allreduce_min(std::uint64_t elem) {
    MPI_Allreduce(MPI_IN_PLACE, &elem, 1, MPI_UINT64_T, MPI_MIN,
                  MPI_COMM_WORLD);
    return elem;
  }
  std::uint64_t allreduce_max(std::uint64_t elem) {
    MPI_Allreduce(MPI_IN_PLACE, &elem, 1, MPI_UINT64_T, MPI_MAX,
                  MPI_COMM_WORLD);
    return elem;
  }
  std::vector<SentRecvId> gather(const std::vector<SentRecvId>& data) {
    mpi::MPIContext ctx;
    std::vector<SentRecvId> recv_buffer(
        ctx.rank() == 0 ? data.size() * ctx.size() : 0ull);
    mpi::TypeMapper<SentRecvId> tm;
    MPI_Gather(data.data(), data.size(), tm.get_mpi_datatype(),
               recv_buffer.data(), data.size(), tm.get_mpi_datatype(), 0,
               ctx.communicator());
    return recv_buffer;
  }

  void reset() { *this = CommunicationVolumeTracker{}; }

  template <typename It> AnalysisResult analyse(It begin, It end) {
    AnalysisResult result;
    for (It it = begin; it != end; ++it) {
      // std::cout << *it << std::endl;
      result.max_sent_bytes =
          std::max(result.max_sent_bytes, it->num_sent_bytes);
      result.max_recv_bytes =
          std::max(result.max_recv_bytes, it->num_recv_bytes);
      result.sum_sent_bytes += it->num_sent_bytes;
      result.sum_recv_bytes += it->num_recv_bytes;
    }
    result.max_io_bytes =
        std::max(result.max_sent_bytes, result.max_recv_bytes);
    // std::cout << result << "\n" << std::endl;
    return result;
  }
  template <typename It> MetaAnalysisResult meta_analyse(It begin, It end) {
    MetaAnalysisResult result;
    for (It it = begin; it != end; ++it) {
      result.max_sent_bytes =
          std::max(result.max_sent_bytes, it->max_sent_bytes);
      result.max_recv_bytes =
          std::max(result.max_recv_bytes, it->max_recv_bytes);
      result.sum_sent_bytes += it->sum_sent_bytes;
      result.sum_recv_bytes += it->sum_recv_bytes;
      result.sum_max_sent_bytes += it->max_sent_bytes;
      result.sum_max_recv_bytes += it->max_recv_bytes;
    }
    result.max_io_bytes =
        std::max(result.max_sent_bytes, result.max_recv_bytes);
    result.sum_max_io_bytes =
        std::max(result.sum_max_sent_bytes, result.sum_max_recv_bytes);
    return result;
  }

  MetaAnalysisResult collect() {
    mpi::MPIContext ctx;

    std::vector<std::string> local_keys;
    std::vector<SentRecvId> local_data;
    for (const auto& [key, value] : id_measurment) {
      local_keys.push_back(key);
      local_data.push_back(value);
    }
    const std::size_t max_num_keys = allreduce_max(local_keys.size());
    const std::size_t min_num_keys = allreduce_min(local_keys.size());
    if (max_num_keys != min_num_keys && ctx.rank() == 0) {
      std::cout << "number keys not equal min: " << min_num_keys
                << " max: " << max_num_keys << std::endl;
      std::abort();
    }
    auto data = gather(local_data);
    // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(local_data););
    // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(data););
    std::vector<AnalysisResult> analysis_results;
    if (ctx.rank() == 0) {
      for (std::size_t i = 0; i < max_num_keys; ++i) {
        std::vector<SentRecvId> data_from_collective;
        for (std::size_t j = 0; j < ctx.size(); ++j) {
          data_from_collective.push_back(data[i + (max_num_keys * j)]);
        }
        analysis_results.push_back(
            analyse(data_from_collective.begin(), data_from_collective.end()));
        // std::cout << analysis_results.back() << std::endl;
      }
      auto meta_analysis =
          meta_analyse(analysis_results.begin(), analysis_results.end());
      return meta_analysis;
    } else {
      return MetaAnalysisResult{};
    }
  }
  std::string output() {
    mpi::MPIContext ctx;
    auto result = collect();
    std::stringstream sstream;
    if (ctx.rank() == 0) {
      sstream << " sum_max_io_bytes=" << result.sum_max_io_bytes
              << " sum_sent_bytes=" << result.sum_sent_bytes
              << " sum_recv_bytes=" << result.sum_recv_bytes;
    }
    return sstream.str();
  }

private:
  std::size_t tracking_number = 0;
  std::unordered_map<std::string, SentRecvId> id_measurment;
};

inline CommunicationVolumeTracker& get_communication_tracker() {
  static CommunicationVolumeTracker tracker;
  return tracker;
}

} // namespace hybridMST
