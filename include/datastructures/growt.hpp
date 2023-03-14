#pragma once

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
}  // namespace internal

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

template <typename Map>
struct Handle {
  Handle(Map& map) : handle(map.get_handle()) {}
  growt::GlobalVIdMap<VId>::handle_type handle;
};
}  // namespace hybridMST::growt
