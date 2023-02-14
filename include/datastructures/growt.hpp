#pragma once

#include <tbb/enumerable_thread_specific.h>

#include <type_traits>

#include "definitions.hpp"
#include "growt/allocator/alignedallocator.hpp"
#include "growt/data-structures/table_config.hpp"
#include "growt/utils/hash/murmur2_hash.hpp"

// from https://github.com/DanielSeemaier/KaMinPar/
namespace hybridMST::growt {
using DefaultHasherType = utils_tm::hash_tm::murmur2_hash;
using DefaultAllocatorType = ::growt::AlignedAllocator<>;

namespace internal {
// workaround 32 bit value bug in growt
template <typename Type>
using Ensure64BitType =
    std::conditional_t<std::numeric_limits<Type>::is_signed, int64_t, uint64_t>;
} // namespace internal

template <typename Value>
using GlobalVIdMap = typename ::growt::table_config<
    VId, internal::Ensure64BitType<Value>, DefaultHasherType,
    DefaultAllocatorType, hmod::neutral, hmod::ref_integrity>::table_type;

using StaticGhostNodeMapping =
    typename ::growt::table_config<VId, VId, DefaultHasherType,
                                   DefaultAllocatorType>::table_type;

template <typename K, typename V, typename Map>
decltype(auto) insert(Map& map, const K& key, const V& value) {
  return map.insert(key + 1, value);
}

template <typename K, typename Map>
decltype(auto) find(Map& map, const K& key) {
  return map.find(key + 1);
}

//template <typename Handle, typename K> decltype(auto) find(Handle& h, K key) {
//  const uint64_t threshold = 100'000'000;
//  if (key > threshold)
//    std::cout << "illegal: " << key << std::endl;
//  return h.find(key + 1);
//}
//template <typename Handle, typename K, typename V>
//decltype(auto) insert(Handle& h, K key, V v) {
//  const uint64_t threshold = 100'000'000;
//  if (key > threshold || v > threshold)
//    std::cout << "illegal: " << key << std::endl;
//  return h.insert(key + 1, v + 1);
//}

template <typename Map> struct Handle {
  Handle(Map& map) : handle(map.get_handle()) {}
  growt::GlobalVIdMap<VId>::handle_type handle;
};
template <typename Map> auto create_handle_ets(Map& map) {
  return tbb::enumerable_thread_specific<growt::GlobalVIdMap<VId>::handle_type>{
      [&]() { return map.get_handle(); }};
}
} // namespace hybridMST::growt
