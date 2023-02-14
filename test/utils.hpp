#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mpi/context.hpp"
#include "util/macros.hpp"

namespace hybridMST::tests {

template <typename T> struct SendData {
  std::vector<std::uint64_t> send_counts;
  std::vector<T> send_data;
};

template <typename T>
inline bool contains_same_elements(std::vector<T> lhs, std::vector<T> rhs) {
  std::sort(lhs.begin(), lhs.end());
  std::sort(rhs.begin(), rhs.end());
  return lhs == rhs;
}

} // namespace hybridMST::tests
