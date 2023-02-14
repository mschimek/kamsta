#pragma once
#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "util/timer.hpp"
#include "util/utils.hpp"


namespace hybridMST {
// https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
template <class Tp> inline void doNotOptimizeAway(Tp const& value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp> inline void doNotOptimizeAway(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

template <typename T> struct PartitionedData {
  PartitionedData(std::size_t data_size, std::size_t nb_partitions)
      : partition_end_idx(nb_partitions, 0), data(data_size) {}
  std::vector<std::size_t> partition_end_idx;
  std::vector<T> data;
  int get_partition(size_t idx) const {
    return idx % partition_end_idx.size();
    // const auto it = std::upper_bound(partition_end_idx.begin(),
    //                                  partition_end_idx.end(), idx);

    // const auto dist = std::distance(partition_end_idx.begin(), it);
    // return dist;
  }
};

template <typename T>
PartitionedData<T> inline get_partitioned_data(size_t n,
                                               hybridMST::mpi::MPIContext ctx) {
  static std::mt19937 gen(ctx.rank());
  std::uniform_int_distribution<std::uint64_t> start_indices_dist(0, n);
  std::uniform_int_distribution<T> data_distribution(0, 255);

  PartitionedData<T> part_data(n, ctx.size());
  std::generate(part_data.partition_end_idx.begin(),
                part_data.partition_end_idx.end(),
                [&]() { return start_indices_dist(gen); });
  std::sort(part_data.partition_end_idx.begin(),
            part_data.partition_end_idx.end());
  if (!part_data.partition_end_idx.empty())
    part_data.partition_end_idx.back() = n;

  std::generate(part_data.data.begin(), part_data.data.end(),
                [&]() { return data_distribution(gen); });
  return part_data;
}



template <typename F>
void benchmark_local(F&& f, std::size_t nb_iterations,
                     const std::string& desc) {
  std::vector<double> intervals;
  for (std::size_t i = 0; i < nb_iterations; ++i) {
    auto t1 = hybridMST::now();
    f();
    auto t2 = hybridMST::now();
    intervals.emplace_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count());
  }
  const auto mean = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
                    intervals.size();
  std::cout << "Execution of " << desc << "\n\tmean running time "
            << "(" << nb_iterations << " iterations):\t" << mean << std::endl;
}

template <typename Enum> struct EnumMapper {
  using EnumString = std::pair<Enum, std::string>;
  EnumMapper(const std::initializer_list<EnumString>& list) {
    for (const auto& elem : list)
      enum_strings.emplace_back(elem);
  }
  Enum get_enum(const std::string& str) const {
    const auto it = std::find_if(enum_strings.begin(), enum_strings.end(),
                                 [&](const EnumString& enum_string) {
                                   return enum_string.second == str;
                                 });
    if (it == enum_strings.end()) {
      hybridMST::mpi::MPIContext ctx;
      ctx.abort(" string " + str + " not found ");
    }
    return it->first;
  }

  std::string get_string(const Enum& enum_) const {
    const auto it = std::find_if(enum_strings.begin(), enum_strings.end(),
                                 [&](const EnumString& enum_string) {
                                   return enum_string.first == enum_;
                                 });
    if (it == enum_strings.end())
      return "UNKNOWN";
    return it->second;
  }
  std::vector<std::pair<Enum, std::string>> enum_strings;
};

} // namespace hybridMST
