#pragma once

#include <mpi/context.hpp>
#include <random>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace hybridMST {
template <std::size_t k, typename Key, typename Value> class LookupMap {
public:
  using HashMap = absl::flat_hash_map<Key, Value>;

  LookupMap() : maps(k) {}

  template <typename Container, typename GetKey, typename GetValue>
  void insert(Container&& key_values, GetKey&& get_key, GetValue&& get_value) {
    mpi::MPIContext ctx;
    clear();
    std::cout << "k: " << k << std::endl;
    SEQ_EX(
        ctx, for (const auto& elem
                  : key_values) {
          std::cout << get_key(elem) << " " << get_value(elem) << std::endl;
        };);
    if constexpr (k == 1) {
      for (std::size_t i = 0; i < key_values.size(); ++i) {
        const auto& elem = key_values[i];
        maps[0].emplace(get_key(elem), get_value(elem));
      }

    } else {
      splitters = determine_splitter(key_values, get_key);

      SEQ_EX(ctx, PRINT_VECTOR(splitters););

      //#pragma omp parallel for schedule(static)
      for (std::size_t i = 0; i < key_values.size(); ++i) {
        const auto& elem = key_values[i];
        const std::size_t map_idx = get_map_index(get_key(elem));
        maps[map_idx].emplace(get_key(elem), get_value(elem));
      }
      ctx.execute_in_order([&]() {
        for (std::size_t i = 0; i < k; ++i) {
          std::cout << "map: " << i << std::endl;
          for (const auto& [key, value] : maps[i]) {
            std::cout << key << " " << value << std::endl;
          }
        }
      });
    }
  }

  Value get(const Key& key) const {
    mpi::MPIContext ctx;
    if constexpr (k == 1) {
      if (maps[0].find(key) == maps[0].end()) {
        std::cout << "not found on rank " << ctx.rank() << " key: " << key
                  << std::endl;
      }
      return maps[0].find(key)->second;
    } else {
      const std::size_t map_idx = get_map_index(key);
      return maps[map_idx].find(key)->second;
    }
  }

  void clear() {
    for (std::size_t i = 0; i < maps.size(); ++i)
      maps[i].clear();
    splitters.clear();
  }

private:
  std::vector<std::size_t> get_splitter_order() const {
    switch (k) {
    case 1:
      return {};
    case 2:
      return {1};
    case 4:
      return {2, 1, 3};
    case 8:
      return {4, 2, 6, 1, 3, 5, 7};
    case 16:
      return {8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    }
    return {};
  }
  template <typename Container, typename GetKey>
  std::vector<Key> determine_splitter(const Container& key_values,
                                      GetKey&& get_key) const {
    constexpr std::size_t oversampling_factor = 20;
    constexpr std::size_t num_samples = oversampling_factor * k;
    std::mt19937 gen(0);
    std::uniform_int_distribution<> dist(0, key_values.size());
    std::vector<Key> samples(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
      samples[i] = get_key(key_values[dist(gen)]);
    }
    std::sort(samples.begin(), samples.end());
    std::vector<Key> splitters(k);
    const std::vector<std::size_t> splitter_order =
        get_splitter_order(); // zero-indexed
    for (std::size_t i = 1; i < splitters.size(); ++i) {
      const auto splitter_order_idx = splitter_order[i - 1];
      splitters[i] = samples[splitter_order_idx * k];
    }
    return splitters;
  }
  std::size_t get_map_index(const Key& key) const {
    constexpr std::size_t log_k = std::log2(k);
    std::size_t j = 1;
    for (std::size_t i = 0; i < log_k; ++i) {
      j = 2 * j + static_cast<std::size_t>(splitters[j] > key);
    }
    j -= k;
    return j;
  }
  void clear_n(std::size_t n) {
    for (std::size_t i = 0; i < n && i < maps.size(); ++i) {
      maps[i].clear();
    }
  }

private:
  std::vector<HashMap> maps;
  std::vector<Key> splitters;
};

template <typename Key, typename Value> struct LookupMapManager {

  LookupMapManager(std::size_t k_) : k{k_} {}
  template <typename Container, typename GetKey, typename GetValue>
  void insert(Container&& key_values, GetKey&& get_key, GetValue&& get_value) {
    switch (k) {
    case 1:
      map_1.insert(std::forward<Container>(key_values),
                   std::forward<GetKey>(get_key),
                   std::forward<GetValue>(get_value));
      break;
    case 2:
      map_2.insert(std::forward<Container>(key_values),
                   std::forward<GetKey>(get_key),
                   std::forward<GetValue>(get_value));
      break;
    case 4:
      map_4.insert(std::forward<Container>(key_values),
                   std::forward<GetKey>(get_key),
                   std::forward<GetValue>(get_value));
      break;
    case 8:
      map_8.insert(std::forward<Container>(key_values),
                   std::forward<GetKey>(get_key),
                   std::forward<GetValue>(get_value));
      break;
    case 16:
      map_16.insert(std::forward<Container>(key_values),
                    std::forward<GetKey>(get_key),
                    std::forward<GetValue>(get_value));
      break;
    }
  }

  Value get(const Key& key) {
    static Value default_value{};
    switch (k) {
    case 1:
      return map_1.get(key);
    case 2:
      return map_2.get(key);
    case 4:
      return map_4.get(key);
    case 8:
      return map_8.get(key);
    case 16:
      return map_16.get(key);
    }
  }

  void clear() {
    switch (k) {
    case 1:
      return map_1.clear();
    case 2:
      return map_2.clear();
    case 4:
      return map_4.clear();
    case 8:
      return map_8.clear();
    case 16:
      return map_16.clear();
    }
  }

  std::size_t k;
  LookupMap<1, Key, Value> map_1;
  LookupMap<2, Key, Value> map_2;
  LookupMap<4, Key, Value> map_4;
  LookupMap<8, Key, Value> map_8;
  LookupMap<16, Key, Value> map_16;
};

} // namespace hybridMST
