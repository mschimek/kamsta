#pragma once

#include <iostream>
#include <unordered_map>

#include "util/macros.hpp"

namespace hybridMST {
struct LocalTimer {
  void start(const std::string& key) {
    REORDERING_BARRIER
    start_ = std::chrono::steady_clock::now();
    REORDERING_BARRIER
    key_ = key;
  }
  void stop() {
    REORDERING_BARRIER
    auto stop_ = std::chrono::steady_clock::now();
    REORDERING_BARRIER

    std::chrono::duration<double> diff = (stop_ - start_);
    double d = diff.count();
    key_value[key_] += d;
  }
  void output(const std::string& prefix) {
    for (const auto& [key, value] : key_value) {
      std::cout << prefix << ": " << key << " " << value << std::endl;
    }
  }
  void reset() { key_value.clear(); }
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string key_;

  std::unordered_map<std::string, double> key_value;
};

inline LocalTimer& lotimer() {
  static LocalTimer t;
  return t;
}
} // namespace hybridMST
