#include "util/timer.hpp"

namespace hybridMST {
std::chrono::high_resolution_clock::time_point now() {
  return std::chrono::high_resolution_clock::now();
}
std::vector<char> StandardKey::serialize() const {
  std::size_t size = id.size();
  const std::size_t necessary_size = sizeof(size) + size + sizeof(count);
  std::vector<char> serialization(necessary_size);
  std::size_t offset = 0;
  if (serialization.size() < sizeof(size)) {
    std::cerr << "GCC was right with its false positive" << std::endl;
    std::terminate();
  }
  std::memcpy(serialization.data(), &size, sizeof(size));
  offset += sizeof(size);
  std::memcpy(serialization.data() + offset, id.data(), id.size());
  offset += id.size();
  std::memcpy(serialization.data() + offset, &count, sizeof(count));
  return serialization;
}
StandardKey deserialize(const char* buf) {
  std::size_t length = 0;
  std::memcpy(&length, buf, sizeof(length));
  StandardKey key{std::string(length, '0'), 0};
  std::size_t offset = sizeof(length);
  for (char& c : key.id) {
    std::memcpy(&c, buf + offset, 1);
    ++offset;
  }
  std::memcpy(&key.count, buf + offset, sizeof(key.count));
  return key;
}
std::vector<char> serialize(std::vector<StandardKey>& keys) {
  std::vector<char> serialization;
  for (const auto& key : keys) {
    const auto& local_serialization = key.serialize();
    serialization.insert(serialization.end(), local_serialization.begin(),
                         local_serialization.end());
  }
  return serialization;
}

std::vector<StandardKey> deserialize(std::vector<char>& serialization) {
  std::vector<StandardKey> keys;
  std::size_t offset = 0;
  do {
    const StandardKey key = deserialize(serialization.data() + offset);
    keys.push_back(key);
    offset += sizeof(std::size_t) + key.id.size() + sizeof(key.count);
  } while (offset < serialization.size());
  return keys;
}

bool operator<(const StandardKey& lhs, const StandardKey& rhs) {
  return std::tie(lhs.id, lhs.count) < std::tie(rhs.id, rhs.count);
}

std::string Timer::get_string(const Timer::DatapointsOperation& op) const {
  switch (op) {
  case DatapointsOperation::SUM:
    return "SUM";
  case DatapointsOperation::MAX_DIF:
    return "MAX_DIF";
  case DatapointsOperation::MAX:
    return "MAX";
  case DatapointsOperation::ID:
    return "ID";
  default:
    return "UNKNOWN";
  };
}

const std::set<Timer::DatapointsOperation>& Timer::all_operations() const {
  const static std::set<DatapointsOperation> set{DatapointsOperation::SUM,
                                                 DatapointsOperation::MAX_DIF,
                                                 DatapointsOperation::MAX};
  return set;
}
void Timer::reset() {
  keyToStart.clear();
  keyToStop.clear();
  keyToTime.clear();
  keyToData.clear();
  phase_to_counters.clear();
  active_phase = default_phase;
};
void Timer::start(const typename Key::Id& key_id,
                  const typename Key::Count& count) {
  asm volatile("" ::: "memory");
  if (is_debug_output_enabled && ctx.rank() == 0)
    std::cout << "start: " << key_id << "-" << count << std::endl;
  if (!measurementEnabled)
    return;
  if (is_barrier_enabled)
    ctx.barrier();
  if (is_debug_output_enabled && ctx.rank() == 0)
    std::cout << "\t after barrier: start: " << key_id << "-" << count
              << std::endl;
  Key key{key_id, count};
  const auto& [itToStart, ignore] = keyToStart.emplace(key, PointInTime());

  // Start measurement
  const PointInTime start = MPI_Wtime();

  itToStart->second = start;
  asm volatile("" ::: "memory");
}

void Timer::start_local(const typename Key::Id& key_id,
                        const typename Key::Count& count) {
  const bool old_barrier_value = is_barrier_enabled;
  set_barrier_value(false);
  start(key_id, count);
  set_barrier_value(old_barrier_value);
}

void Timer::start(const typename Key::Id& key_id) { start(key_id, 0u); }
void Timer::start_phase_measurement(const typename Key::Id& key_id) {
  if (!measurementEnabled)
    return;
  const auto& [time_count, add_count] = phase_to_counters[active_phase];
  start(active_phase + "_" + key_id, time_count);
}
void Timer::stop_phase_measurement(const typename Key::Id& key_id) {
  if (!measurementEnabled)
    return;
  auto& [time_count, add_count] = phase_to_counters[active_phase];
  stop(active_phase + "_" + key_id, time_count);
  ++time_count;
}

void Timer::add(const typename Key::Id& key_id,
                const typename Key::Count& count, std::int64_t value,
                const DatapointsOperation op) {
  if (!measurementEnabled)
    return;
  if (is_debug_output_enabled && ctx.rank() == 0)
    std::cout << "add: " << key_id << "-" << count << " op: " << get_string(op)
              << std::endl;
  Key key{key_id, count};
  key.append_to_id(get_string(op));
  keyToData.emplace(key, std::make_pair(value, op));
}
void Timer::add(const typename Key::Id& key_id,
                const typename Key::Count& count, std::int64_t value,
                const std::set<DatapointsOperation>& ops) {
  for (const auto& op : ops) {
    add(key_id, count, value, op);
  }
}

void Timer::add_phase(const typename Key::Id& key_id,
                      const typename Key::Count& count, std::int64_t value,
                      const std::set<DatapointsOperation>& ops) {
  for (const auto& op : ops) {
    add(active_phase + "_" + key_id, count, value, op);
  }
}

void Timer::stop(const typename Key::Id& key_id,
                 const typename Key::Count& count) {
  asm volatile("" ::: "memory");
  if (is_debug_output_enabled && ctx.rank() == 0)
    std::cout << "stop: " << key_id << "-" << count << std::endl;
  if (!measurementEnabled)
    return;

  const PointInTime endPoint = MPI_Wtime();
  if (is_barrier_enabled)
    ctx.barrier();
  const PointInTime endPointAfterBarrier = MPI_Wtime();

  // check whether key is present
  const Key key{key_id, count};
  const auto it = keyToStart.find(key);
  if (it == keyToStart.end()) {
    std::cout << "Key: " << key << " has no corresponding start" << std::endl;
    std::abort();
  }
  const PointInTime startPoint = it->second;

  const TimeIntervalDataType elapsedActiveTime =
      static_cast<TimeIntervalDataType>((endPoint - startPoint) * 1000000.0);
  const TimeIntervalDataType elapsedTotalTime =
      static_cast<TimeIntervalDataType>((endPointAfterBarrier - startPoint) *
                                        1000000.0);

  keyToStop.emplace(key, endPoint);
  keyToTime.emplace(key, Timeintervals{elapsedActiveTime, elapsedTotalTime});
  asm volatile("" ::: "memory");
}

void Timer::stop_local(const typename Key::Id& key_id,
                       const typename Key::Count& count) {
  const bool old_barrier_value = is_barrier_enabled;
  set_barrier_value(false);
  stop(key_id, count);
  set_barrier_value(old_barrier_value);
}

std::string Timer::output(std::string prefix) const {
  std::vector<TimeOutputType> time_measurements;
  time_measurements.reserve(keyToTime.size());
  collect(std::back_inserter(time_measurements));

  // debug output: data from each PE
  // std::vector<TimeOutputType> debug_time_measurements;
  // time_measurements.reserve(keyToTime.size() *
  // static_cast<std::size_t>(ctx.size()));
  // collect_all_data(std::back_inserter(debug_time_measurements));

  // std::vector<StartStopOutputType> start_stop_time_measurements;
  // time_measurements.reserve(keyToTime.size() *
  // static_cast<std::size_t>(ctx.size()));
  // collect_start_stop(std::back_inserter(start_stop_time_measurements));

  std::vector<DatapointOutputType> datapoints;
  datapoints.reserve(keyToData.size());
  collect_datapoints(std::back_inserter(datapoints));
  std::stringstream sstream;

  if (ctx.rank() == 0) {
    sstream << "RESULT " << prefix << " ";
    for (const auto& elem : time_measurements)
      sstream << print(elem) << " ";
    for (const auto& elem : datapoints)
      sstream << print(elem) << " ";
  }
  return sstream.str();
}

} // namespace hybridMST
