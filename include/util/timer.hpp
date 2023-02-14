#pragma once

#include <chrono>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "mpi/allgather.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/context.hpp"

namespace hybridMST {
std::chrono::high_resolution_clock::time_point now();

struct StandardKey {
  using Id = std::string;
  using Count = std::uint64_t;
  Id id;
  Count count;
  StandardKey() = default;
  StandardKey(const Id& id, const Count& count) : id{id}, count{count} {}
  void append_to_id(const std::string& str) {
    id.append("_");
    id.append(str);
  }
  bool operator==(const StandardKey& other_key) const {
    return std::tie(id, count) == std::tie(other_key.id, other_key.count);
  }
  friend std::ostream& operator<<(std::ostream& out, const StandardKey& key) {
    return out << key.id << "-" << key.count;
  }
  std::vector<char> serialize() const;
};
} // namespace hybridMST

namespace std {
template <> struct hash<hybridMST::StandardKey> {
  std::size_t operator()(const hybridMST::StandardKey& key) const {
    return (hash<hybridMST::StandardKey::Id>{}(key.id) ^
            (std::hash<hybridMST::StandardKey::Count>{}(key.count) << 1) >> 1);
  }
};
} // namespace std
namespace hybridMST {
StandardKey deserialize(const char* buf);
std::vector<char> serialize(std::vector<StandardKey>& keys);
std::vector<StandardKey> deserialize(std::vector<char>& serialization);
bool operator<(const StandardKey& lhs, const StandardKey& rhs);

class Timer {
  using PointInTime = double;
  using TimeIntervalDataType = unsigned long long;
  using Key = StandardKey;

  struct Timeintervals {
    TimeIntervalDataType activeTime;
    TimeIntervalDataType totalTime;
    friend std::ostream& operator<<(std::ostream& out,
                                    const Timeintervals& intervals) {
      return out << "active time: " << intervals.activeTime
                 << " total time: " << intervals.totalTime;
    }
  };
  struct TimeType {
    TimeIntervalDataType time;
    std::string type;
  };

  struct DataType {
    std::int64_t data;
    std::string type;
  };

  struct StartStop {
    PointInTime start;
    PointInTime stop;
  };

public:
  enum class DatapointsOperation { SUM, MAX_DIF, MAX, ID };
  std::string get_string(const DatapointsOperation& op) const;

  const std::set<DatapointsOperation>& all_operations() const;

  using TimeOutputType = std::pair<Key, TimeType>;
  using StartStopOutputType = std::tuple<Key, PointInTime, PointInTime>;
  using DatapointOutputType = std::pair<Key, std::string>;
  Timer() = default;

  void reset();

  void start(const typename Key::Id& key_id, const typename Key::Count& count);
  void start_local(const typename Key::Id& key_id,
                   const typename Key::Count& count);
  void start_local(const typename Key::Id& key_id) { start_local(key_id, 0u); }
  void start(const typename Key::Id& key_id);
  void start_phase_measurement(const typename Key::Id& key_id);
  void stop_phase_measurement(const typename Key::Id& key_id);

  void add(const typename Key::Id& key_id, const typename Key::Count& count,
           std::int64_t value, const DatapointsOperation op);
  void add(const typename Key::Id& key_id, const typename Key::Count& count,
           std::int64_t value, const std::set<DatapointsOperation>& ops);

  void add_phase(const typename Key::Id& key_id,
                 const typename Key::Count& count, std::int64_t value,
                 const std::set<DatapointsOperation>& ops);

  void increment_phase_add_count() {
    auto& [_, add_counter] = phase_to_counters[active_phase];
    ++add_counter;
  }
  const Key::Count& get_phase_add_count() {
    const auto& [_, add_counter] = phase_to_counters[active_phase];
    return add_counter;
  }

  void stop(const typename Key::Id& key_id, const typename Key::Count& count);
  void stop(const typename Key::Id& key_id) { stop(key_id, 0u); }
  void stop_local(const typename Key::Id& key_id,
                  const typename Key::Count& count);
  void stop_local(const typename Key::Id& key_id) { stop_local(key_id, 0u); }
  template <typename OutputIterator> void collect(OutputIterator out) const {
    std::vector<Key> local_keys;
    if (ctx.rank() == 0) {
      for (const auto& [key, ignore] : keyToTime)
        local_keys.push_back(key);
    }
    std::sort(local_keys.begin(), local_keys.end());
    std::vector<char> serialization = serialize(local_keys);
    std::vector<char> root_serialization = mpi::allgatherv(serialization);
    std::vector<Key> root_keys = deserialize(root_serialization);
    for (const auto& key : root_keys) {
      const TimeType tt{maxTime(key), "maxTime"};
      out = std::make_pair(key, tt);
    }

    /*value.setType("avgTime");
    value.setValue(avgTime(key));
    out = std::make_pair(key, value);

    value.setType("maxTime");
    value.setValue(maxTime(key));
    out = std::make_pair(key, value);

    value.setType("minTime");
    value.setValue(minTime(key));
    out = std::make_pair(key, value);

    value.setType("avgLoss");
    value.setValue(avgLoss(key));
    out = std::make_pair(key, value);

    value.setType("maxLoss");
    value.setValue(maxLoss(key));
    out = std::make_pair(key, value);

    value.setType("minLoss");
    value.setValue(minLoss(key));
    out = std::make_pair(key, value);*/
  }
  template <typename OutputIterator>
  void collect_all_data(OutputIterator out) const {
    std::vector<Key> local_keys;
    if (ctx.rank() == 0) {
      for (const auto& [key, ignore] : keyToTime)
        local_keys.push_back(key);
    }
    std::sort(local_keys.begin(), local_keys.end());
    std::vector<char> serialization = serialize(local_keys);
    std::vector<char> root_serialization = mpi::allgatherv(serialization);
    std::vector<Key> root_keys = deserialize(root_serialization);
    for (const auto& key : root_keys) {
      const auto& all_times = allTimes(key);
      for (std::size_t i = 0; i < all_times.size(); ++i) {
        auto key_ = key;
        std::string p_info = "p:" + std::to_string(i) + "-";
        key_.id.insert(0, p_info);
        const auto& time = all_times[i];
        const TimeType tt{time, "allTimes"};
        out = std::make_pair(key_, tt);
      }
    }
  }

  template <typename OutputIterator>
  void collect_start_stop(OutputIterator out) const {
    std::vector<Key> local_keys;
    if (ctx.rank() == 0) {
      for (const auto& [key, ignore] : keyToTime)
        local_keys.push_back(key);
    }
    std::sort(local_keys.begin(), local_keys.end());
    std::vector<char> serialization = serialize(local_keys);
    std::vector<char> root_serialization = mpi::allgatherv(serialization);
    std::vector<Key> root_keys = deserialize(root_serialization);
    for (const auto& key : root_keys) {
      const auto& all_times = all_start_stop_point_in_time(key);
      for (std::size_t i = 0; i < all_times.size(); ++i) {
        auto key_ = key;
        std::string p_info = "p:" + std::to_string(i) + "-";
        key_.id.insert(0, p_info);
        const auto& time = all_times[i];
        out = std::make_tuple(key_, time.start, time.stop);
      }
    }
  }

  template <typename OutputIterator>
  void collect_datapoints(OutputIterator out) const {
    for (const auto& [key, data_op] : keyToData) {
      auto [datapoint, operation] = data_op;
      auto data = mpi::allgather(datapoint);
      switch (operation) {
      case DatapointsOperation::SUM:
        out = std::make_pair(key, std::to_string(std::accumulate(
                                      data.begin(), data.end(), 0ll)));
        break;
      case DatapointsOperation::MAX_DIF: {
        auto [it_min, it_max] = std::minmax_element(data.begin(), data.end());
        out = std::make_pair(key, std::to_string(*it_max - *it_min));
      } break;
      case DatapointsOperation::MAX: {
        auto [it_min, it_max] = std::minmax_element(data.begin(), data.end());
        out = std::make_pair(key, std::to_string(*it_max));
      } break;
      case DatapointsOperation::ID: {
        std::stringstream ss;
        for (const auto& elem : data)
          ss << elem << ":";
        std::string str = ss.str();
        if (!str.empty())
          str.pop_back();
        out = std::make_pair(key, str);
      } break;
      };
    }
  }

  std::string output(std::string prefix = "") const;

  void set_barrier_value(bool value) { is_barrier_enabled = value; }
  void set_debug_output_enablement(bool value) {
    is_debug_output_enabled = value;
  }

  void start_phase(const std::string& phase_id) {
    active_phase = phase_id;
    if (phase_to_counters.find(phase_id) == phase_to_counters.end())
      phase_to_counters.emplace(phase_id, std::make_pair(0ull, 0ull));
  }

  void stop_phase() { active_phase = default_phase; }
  void enable_measurements() { measurementEnabled = true; }
  void disable_measurements() { measurementEnabled = false; }

private:
  bool measurementEnabled = true;
  bool is_barrier_enabled = true;
  bool is_debug_output_enabled = false;
  hybridMST::mpi::MPIContext ctx;

  std::string print(const Timer::TimeOutputType& elem) const {
    std::stringstream ss;
    ss << elem.first << "=" << elem.second.time;
    return ss.str();
  }

  std::string print(const Timer::StartStopOutputType& elem) const {
    std::stringstream ss;
    ss << std::get<0>(elem) << "=" << std::get<1>(elem) << "="
       << std::get<2>(elem);
    return ss.str();
  }

  std::string print(const Timer::DatapointOutputType& elem) const {
    std::stringstream ss;
    ss << elem.first << "=" << elem.second;
    return ss.str();
  }
  TimeIntervalDataType maxTime(const Key& key) const {
    const auto itKeyTime = keyToTime.find(key);
    const bool is_present = itKeyTime != keyToTime.end();
    const auto& [active_time, total_time] =
        is_present ? itKeyTime->second : Timeintervals{0ull, 0ull};
    return hybridMST::mpi::allreduce_max(active_time);
  }
  std::vector<TimeIntervalDataType> allTimes(const Key& key) const {
    const auto itKeyTime = keyToTime.find(key);
    const bool is_present = itKeyTime != keyToTime.end();
    const auto& [active_time, total_time] =
        is_present ? itKeyTime->second : Timeintervals{0ull, 0ull};
    return hybridMST::mpi::allgather(active_time);
  }

  std::vector<StartStop> all_start_stop_point_in_time(const Key& key) const {
    const auto it_start = keyToStart.find(key);
    const auto it_stop = keyToStop.find(key);
    const bool is_start_present = it_start != keyToStart.end();
    const bool is_stop_present = it_stop != keyToStop.end();
    const auto& start_point = is_start_present ? it_start->second : 0;
    const auto& stop_point = is_stop_present ? it_stop->second : 0;
    return hybridMST::mpi::allgather(StartStop{start_point, stop_point});
  }

  std::unordered_map<Key, PointInTime> keyToStart;
  std::unordered_map<Key, PointInTime> keyToStop;
  std::unordered_map<Key, Timeintervals> keyToTime;
  std::unordered_map<Key, std::pair<std::int64_t, DatapointsOperation>>
      keyToData;
  std::unordered_map<std::string, std::pair<std::uint64_t, std::uint64_t>>
      phase_to_counters;
  const std::string default_phase = "default_phase";
  std::string active_phase = default_phase;
};

inline Timer& get_timer() {
  static Timer timer;
  return timer;
}
} // namespace hybridMST
