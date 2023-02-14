#pragma once

#include <abseil/absl/container/flat_hash_map.h>
#include <omp.h>

template <typename Key, typename Value> class SequentialWriteLookUpMap {
public:
  using HashMap = absl::flat_hash_map<Key, Value>;

  template <typename Container, typename GetKey, typename GetValue>
  void insert(Container&& key_values, GetKey&& get_key, GetValue&& get_value) {
    for (std::size_t i = 0; i < key_values.size(); ++i) {
      const auto& elem = key_values[i];
      map.emplace(get_key(elem), get_value(elem));
    }
  }
  void clear() { map.clear(); }
  const Value& get(const Key& key) { return map.find(key)->second; }

private:
  HashMap map;
};

template <typename Key, typename Value> class ParallelWriteLookUpMap {
public:
  using HashMap = absl::flat_hash_map<Key, Value>;

  ParallelWriteLookUpMap(std::size_t num_threads_)
      : num_threads{num_threads_}, maps(num_threads_) {}

  template <typename Container, typename GetKey, typename GetValue>
  void insert(Container&& key_values, GetKey&& get_key, GetValue&& get_value) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < key_values.size(); ++i) {
      std::size_t thread_id = omp_get_thread_num();
      const auto& elem = key_values[i];
      maps[thread_id].emplace(get_key(elem), get_value(elem));
    }
  }
  void clear() {
    for (auto& map : maps) {
      map.clear();
    }
  }
  const Value& get(const Key& key) const {
    static Value default_value{};
    for (std::size_t i = 0; i < num_threads; ++i) {
      auto it = maps[i].find(key);
      if (it != maps[i].end())
        return it->second;
    }
    return default_value;
  }

private:
  std::size_t num_threads;
  std::vector<HashMap> maps;
};
