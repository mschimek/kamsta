#pragma once

#include <atomic>
#include <functional>
#include <utility>

#include "definitions.hpp"

namespace hybridMST {
template <typename Comp = std::less<Weight>>
void write_min(std::atomic<EdgeIdWeight>& a, const EdgeIdWeight& new_a,
               Comp&& comp = Comp{}) {
  static_assert(std::atomic<EdgeIdWeight>::is_always_lock_free,
                "type is not lock free");
  auto curr = a.load();
  while (comp(new_a.weight, curr.weight) &&
         !a.compare_exchange_weak(curr, new_a)) {
  }
}

template <typename Comp>
void write_min(std::atomic<EdgeIdWeightDst>& a, const EdgeIdWeightDst& new_a,
               Comp&& comp = Comp{}) {
  auto curr = a.load();
  while (comp(new_a, curr) &&
         !a.compare_exchange_strong(curr, new_a)) {
  }
}

template <typename Comp = std::less<EdgeIdWeight>>
void write_min_gbbs(std::atomic<EdgeIdWeight>& a, const EdgeIdWeight& new_a,
                    Comp&& comp = Comp{}) {
  static_assert(std::atomic<EdgeIdWeight>::is_always_lock_free,
                "type is not lock free");
  auto curr = a.load();
  while (comp(new_a, curr) && !a.compare_exchange_weak(curr, new_a)) {
  }
}

inline void write_min_weight(std::atomic<Weight>& a, const Weight& new_weight) {
  static_assert(std::atomic<Weight>::is_always_lock_free,
                "type is not lock free");
  auto curr = a.load();
  while ((new_weight < curr) && !a.compare_exchange_weak(curr, new_weight)) {
  }
}
} // namespace hybridMST
