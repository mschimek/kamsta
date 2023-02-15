#pragma once

#include <vector>

// #include "datastructures/concurrent_lookup_map.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/context.hpp"
#include "mpi/grid_communicators.hpp"
#include "util/macros.hpp"
#include "util/utils.hpp"

namespace hybridMST {
template <typename T> class DistributedArray {
public:
  struct IndexValue {
    std::size_t index;
    T value;
    IndexValue() = default;
    IndexValue(std::size_t index_, T value_) : index{index_}, value{value_} {}
    bool operator<(const IndexValue& other) {
      return std::tie(index, value) < std::tie(other.index, other.value);
    }
    bool operator==(const IndexValue& other) {
      return std::tie(index, value) == std::tie(other.index, other.value);
    }
    friend std::ostream& operator<<(std::ostream& out,
                                    const IndexValue& index_value) {
      return out << "(" << index_value.index << ", (" << index_value.value
                 << ")";
    }
  };
  template <typename Init>
  DistributedArray(std::size_t n, Init&& init)
      : n_{n}, remainder_{n_ % ctx_.size()}, small_slice_size_{n_ /
                                                               ctx_.size()} {
    const std::size_t rank = static_cast<std::size_t>(
        ctx_.rank()); // avoid implicit conversion warning
    begin_ = (rank * small_slice_size_) + std::min(rank, remainder_);
    end_ = begin_ + small_slice_size_ +
           static_cast<std::size_t>(rank < remainder_);
    data_.resize(end_ - begin_);
    assign_initialize(data_, [&](std::size_t i) { return init(i, begin_); });
  }
  DistributedArray(std::size_t n)
      : n_{n}, remainder_{n_ % ctx_.size()}, small_slice_size_{n_ /
                                                               ctx_.size()} {
    const std::size_t rank = static_cast<std::size_t>(
        ctx_.rank()); // avoid implicit conversion warning
    begin_ = (rank * small_slice_size_) + std::min(rank, remainder_);
    end_ = begin_ + small_slice_size_ +
           static_cast<std::size_t>(rank < remainder_);
    data_.resize(end_ - begin_);
  }

  PEID get_pe(std::size_t i) const {
    const VId bigger_slice_size = (small_slice_size_ + 1);
    const VId bigger_slice_range = bigger_slice_size * remainder_;
    const bool in_bigger_slice_range = i < bigger_slice_range;
    if (in_bigger_slice_range)
      return i / (bigger_slice_size);
    i -= bigger_slice_range;
    return (i / small_slice_size_) + remainder_;
  }
  decltype(auto) begin() { return data_.begin(); }
  decltype(auto) end() { return data_.end(); }

  std::size_t index_begin() const { return begin_; }
  std::size_t index_end() const { return end_; }
  std::size_t get_local_index(std::size_t global_index) const {
    MPI_ASSERT_((begin_ <= global_index && global_index < end_),
                "global_index: " << global_index << " not in range ");
    return global_index - begin_;
  }

  T& get_value_locally(std::size_t global_index) {
    return data_[get_local_index(global_index)];
  }
  const T& get_value_locally(std::size_t global_index) const {
    return data_[get_local_index(global_index)];
  }

  void set_value_locally(const IndexValue& index_value) {
    const auto local_index = get_local_index(index_value.index);
    data_[local_index] = index_value.value;
  }

  void set_value_locally(const std::size_t& index, const T& value) {
    const auto local_index = get_local_index(index);
    data_[local_index] = value;
  }

  template <typename Container> void set_values(const Container& index_values) {
    mpi::MPIContext ctx;
    auto filter = [&](const IndexValue& elem, const std::size_t&) {
      const PEID pe = get_pe(elem.index);
      if (pe == ctx_.rank()) {
        set_value_locally(elem);
        return true;
      }
      return false;
    };
    auto transformer = [](const IndexValue& elem, const std::size_t&) {
      return elem;
    };
    auto dst_computer = [&](const IndexValue& elem, const std::size_t&) {
      return get_pe(elem.index);
    };
    auto recv =
        mpi::alltoall_combined(index_values, filter, transformer, dst_computer);
    // Assumption: there is only at most one update request for each vertex.
    MPI_ASSERT(ctx_, are_elements_unique(recv.buffer),
               " elements are not unique");
    parallel_for(0, recv.buffer.size(), [&](std::size_t i) {
      const auto& elem = recv.buffer[i];
      set_value_locally(elem);
    });
  }

  template <typename InputContainer>
  non_init_vector<IndexValue>
  get_values_in_vector(const InputContainer& global_indices) const {
    mpi::MPIContext ctx;
    auto filter = False_Predicate{};
    auto transformer = [](const std::size_t& elem, const std::size_t&) {
      return elem;
    };
    auto dst_computer = [&](const std::size_t& elem, const std::size_t&) {
      return get_pe(elem);
    };
    // TODO find better mechanism for request-reply scheme with
    // alltoall-combined
    auto requests = mpi::twopass_alltoallv_openmp_special(
        global_indices, filter, transformer, dst_computer);

    {
      const auto& count = get_timer().get_phase_add_count();
      get_timer().add_phase("send_count", count, global_indices.size(),
                            {Timer::DatapointsOperation::ID});
      get_timer().increment_phase_add_count();
    }

    {
      const auto& count = get_timer().get_phase_add_count();
      get_timer().add_phase("recv_count", count, requests.buffer.size(),
                            {Timer::DatapointsOperation::ID});
      get_timer().increment_phase_add_count();
    }
    auto filter_reply = False_Predicate{};
    auto transformer_reply = [&](const std::size_t& elem, const std::size_t&) {
      return IndexValue{elem, get_value_locally(elem)};
    };
    auto dst_computer_reply = [&](const std::size_t&, const std::size_t& i) {
      return requests.get_pe(i);
    };
    auto reply = mpi::alltoall_combined(requests.buffer, filter_reply,
                                        transformer_reply, dst_computer_reply);
    return std::move(
        reply.buffer); // TODO is move necesary, because of subobject?
  }

  template <typename InputContainer>
  non_init_vector<IndexValue>
  get_values_in_vector_filter(const InputContainer& global_indices) const {
    // assume global indices are already filtered
    mpi::MPIContext ctx;
    const auto& twolevel_comm = mpi::get_grid_communicators();
    auto filter = False_Predicate{};
    auto transformer = [](const std::size_t& elem, const std::size_t&) {
      return elem;
    };
    auto dst_computer_request_first_level = [&](const std::size_t& elem,
                                                const std::size_t&) {
      return twolevel_comm.col_index(get_pe(elem));
    };
    auto requests_first_level = mpi::twopass_alltoallv_openmp_special(
        global_indices, filter, transformer, dst_computer_request_first_level,
        twolevel_comm.get_row_ctx());

    parlay::hashtable<parlay::hash_numeric<VId>> table(
        requests_first_level.buffer.size(), parlay::hash_numeric<VId>{});
    parallel_for(0, requests_first_level.buffer.size(), [&](std::size_t i) {
      const auto& idx = requests_first_level.buffer[i];
      table.insert(idx);
    });
    auto entries = table.entries();
    auto dst_computer_request_second_level = [&](const std::size_t& elem,
                                                 const std::size_t&) {
      return twolevel_comm.row_index(get_pe(elem));
    };
    auto requests_second_level = mpi::twopass_alltoallv_openmp_special(
        entries, filter, transformer, dst_computer_request_second_level,
        twolevel_comm.get_col_ctx());

    auto transformer_reply = [&](const std::size_t& elem, const std::size_t&) {
      return IndexValue{elem, get_value_locally(elem)};
    };
    auto dst_computer_reply = [&](const std::size_t&, const std::size_t& i) {
      return requests_second_level.get_pe(i);
    };
    auto reply_second_level = mpi::twopass_alltoallv_openmp_special(
        requests_second_level.buffer, filter, transformer_reply,
        dst_computer_reply, twolevel_comm.get_col_ctx());
    growt::GlobalVIdMap<T> map(reply_second_level.buffer.size() * 1.25);
    parallel_for(0, reply_second_level.buffer.size(), [&](std::size_t i) {
      const auto& elem = reply_second_level.buffer[i];
      growt::insert(map, elem.index, elem.value);
    });

    auto transformer_reply_first_level = [&](const std::size_t& elem,
                                             const std::size_t&) {
      auto it = map.find(elem + 1);
      return IndexValue{elem, (*it).second};
    };
    auto dst_computer_first_level_reply = [&](const std::size_t&,
                                              const std::size_t& i) {
      return requests_first_level.get_pe(i);
    };
    auto reply_first_level = mpi::twopass_alltoallv_openmp_special(
        requests_first_level.buffer, filter, transformer_reply_first_level,
        dst_computer_first_level_reply, twolevel_comm.get_row_ctx());

    return std::move(reply_first_level.buffer);
  }

  template <typename InputContainer>
  growt::GlobalVIdMap<T>
  get_values(const InputContainer& global_indices) const {
    mpi::MPIContext ctx;
    auto filter = False_Predicate{};
    auto transformer = [](const std::size_t& elem, const std::size_t&) {
      return elem;
    };
    auto dst_computer = [&](const std::size_t& elem, const std::size_t&) {
      return get_pe(elem);
    };
    auto requests = mpi::alltoall_combined(global_indices, filter, transformer,
                                           dst_computer);
    auto filter_reply = False_Predicate{};
    auto transformer_reply = [&](const std::size_t& elem, const std::size_t&) {
      return IndexValue{elem, get_value_locally(elem)};
    };
    auto dst_computer_reply = [&](const std::size_t&, const std::size_t& i) {
      return requests.get_pe(i);
    };
    auto reply = mpi::alltoall_combined(requests.buffer, filter_reply,
                                        transformer_reply, dst_computer_reply);
    growt::GlobalVIdMap<T> map(reply.buffer.size() * 1.25);
    parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
      const auto& elem = reply.buffer[i];
      growt::insert(map, elem.index, elem.value);
    });
    return map;
  }

private:
  mpi::MPIContext ctx_;
  std::size_t n_;         ///< initial global number of vertices
  std::size_t remainder_; ///< number of vertices modulo number of mpi ranks
  std::size_t
      small_slice_size_; ///< number of vertices divided by number of mpi ranks
                         ///< -> actual slice might be +1 as of remainder_
  std::size_t begin_;    ///< first locally managed vertex
  std::size_t end_;      ///< one past the last locally managed vertex
  std::vector<T> data_;  ///< actual parent info
};

template <typename U>
std::ostream& operator<<(std::ostream& out, const DistributedArray<U>& array) {
  out << "begin: " << array.index_begin() << " v_end: " << array.index_end()
      << "\n";
  for (std::size_t i = array.index_begin(); i < array.index_end(); ++i) {
    out << "\t[" << i << "] = " << array.get_value_locally(i) << "\n";
  }
  return out;
}
} // namespace hybridMST
