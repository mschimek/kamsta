#pragma once

#include <cstdint>
#include <exception>
namespace hybridMST {
template <typename OutputIt> auto encode_value(uint64_t v, OutputIt out_it) {
  if (v < 128) {
    *out_it = uint8_t(v);
    ++out_it;
  } else if (v < 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 07) & 0x7F);
    ++out_it;
  } else if (v < 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 14) & 0x7F);
    ++out_it;
  } else if (v < 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 21) & 0x7F);
    ++out_it;
  } else if (v < 128llu * 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 28) & 0x7F);
    ++out_it;
  } else if (v < 128llu * 128 * 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 28) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 35) & 0x7F);
    ++out_it;
  } else if (v < 128llu * 128 * 128 * 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 28) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 35) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 42) & 0x7F);
    ++out_it;
  } else if (v < 128llu * 128 * 128 * 128 * 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 28) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 35) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 42) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 49) & 0x7F);
    ++out_it;
  } else if (v < 128llu * 128 * 128 * 128 * 128 * 128 * 128 * 128 * 128) {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 28) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 35) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 42) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 49) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 56) & 0x7F);
    ++out_it;
  } else {
    *out_it = uint8_t(((v >> 00) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 07) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 14) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 21) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 28) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 35) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 42) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 49) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t(((v >> 56) & 0x7F) | 0x80);
    ++out_it;
    *out_it = uint8_t((v >> 63) & 0x7F);
    ++out_it;
  }
  return out_it;
}

template <typename It> uint64_t decode_value(It& it) {
  std::uint64_t u;
  std::uint64_t v = *it;
  ++it;
  if (!(v & 0x80))
    return v;
  v &= 0x7F;
  u = *it, ++it, v |= (u & 0x7F) << 7;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 14;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 21;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 28;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 35;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 42;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 49;
  if (!(u & 0x80))
    return v;
  u = *it, ++it, v |= (u & 0x7F) << 56;
  if (!(u & 0x80))
    return v;
  u = *it, ++it;
  if (u & 0xFE)
    throw "Overflow during varint64 decoding.";
  v |= (u & 0x7F) << 63;
  return v;
}
} // namespace hybridMST
