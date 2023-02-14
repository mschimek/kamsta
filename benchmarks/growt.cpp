#include <random>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <vector>

#include <type_traits>

#include "definitions.hpp"
#include "growt/allocator/alignedallocator.hpp"
#include "growt/data-structures/table_config.hpp"
#include "growt/utils/hash/murmur2_hash.hpp"

using DefaultHasherType = utils_tm::hash_tm::murmur2_hash;
using DefaultAllocatorType = ::growt::AlignedAllocator<>;
using hashtable =
    ::growt::table_config<uint64_t, uint64_t, DefaultHasherType,
                          DefaultAllocatorType, hmod::growable>::table_type;
template <typename Map> auto create_handle_ets(Map& map) {
  return tbb::enumerable_thread_specific<hashtable::handle_type>{
      [&]() { return map.get_handle(); }};
}

int main() {
  std::size_t n = 5000000;
  std::mt19937 gen;
  std::uniform_int_distribution<std::uint64_t> value_gen(2, 10000);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> data(n);
  for (auto& [src, dst] : data) {
    src = value_gen(gen);
    dst = value_gen(gen);
  }
  auto tmp = data;
  data.insert(data.end(), tmp.begin(), tmp.end());

  hashtable table(n);
  auto handles = create_handle_ets(table);
  ::tbb::parallel_for(tbb::blocked_range<std::size_t>(0, data.size()),
                      [&](tbb::blocked_range<std::size_t> r) {
                        for (std::size_t i = r.begin(); i != r.end(); ++i) {
                          auto& elem = data[i];
                          handles.local().insert(elem.first, elem.second);
                        }
                      });

  auto handle = table.get_handle();
  for (const auto& [key, value] : data) {
    auto it = handle.find(key);
    if (it == handle.end()) {
      std::cout << "(" << key << ", " << value << ") not found" << std::endl;
      std::terminate();
    }
    //if ((*it).second != value) {
    //  std::cout << (*it).second << " instead of "
    //            << "(" << key << ", " << value << ") not found" << std::endl;
    //  std::terminate();
    //}
  }
}
