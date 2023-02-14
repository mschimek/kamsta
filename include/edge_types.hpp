#pragma once
#include <cstdint>
#include <ostream>
#include <utility>

namespace hybridMST {
inline uint64_t set_compound(uint64_t id, uint8_t weight) {
  uint64_t compound = 0;
  compound = weight;
  compound <<= 56;
  compound |= id;
  return compound;
}

inline uint64_t get_edge_id_external(uint64_t compound) {
  compound &= 0x00FFFFFFFFFFFFFF;
  return compound;
}

template <typename WEdgeIdType>
std::ostream& print_wedgeid(std::ostream& out, const WEdgeIdType& e) {
  return out << "(" << e.get_src() << ", " << e.get_dst() << ", " << static_cast<int>(e.get_weight())
      << ", " << e.get_edge_id() << ")";
}

template <typename WEdgeIdType>
std::ostream& print_wedge(std::ostream& out, const WEdgeIdType& e) {
  return out << "(" << e.get_src() << ", " << e.get_dst() << ", " << static_cast<int>(e.get_weight())
      << ")";
}

struct WEdge10 {
  uint16_t src_low;
  uint16_t src_high;
  uint16_t dst_low;
  uint16_t dst_high;
  uint8_t weight;
  uint8_t get_weight() const { return weight; }
  void set_weight(uint8_t weight) { this->weight = weight; }
  uint64_t get_src() const {
    uint64_t src = src_high;
    src <<= 16;
    src |= src_low;
    return src;
  }
  uint64_t get_dst() const {
    uint64_t dst = dst_high;
    dst <<= 16;
    dst |= dst_low;
    return dst;
  }

  void set_src(uint64_t src) {
    this->src_low = src;
    this->src_high = src >> 16;
  }
  void set_dst(uint64_t dst) {
    this->dst_low = dst;
    this->dst_high = dst >> 16;
  }
};

struct WEdge12 {
  uint32_t src_low;
  uint32_t dst_low;
  uint8_t src_high;
  uint8_t dst_high;
  uint8_t weight;
  uint8_t get_weight() const { return weight; }
  void set_weight(uint8_t weight) { this->weight = weight; }
  uint64_t get_src() const {
    uint64_t src = src_high;
    src <<= 32;
    src |= src_low;
    return src;
  }
  uint64_t get_dst() const {
    uint64_t dst = dst_high;
    dst <<= 32;
    dst |= dst_low;
    return dst;
  }

  void set_src(uint64_t src) {
    this->src_low = src;
    this->src_high = src >> 32;
  }
  void set_dst(uint64_t dst) {
    this->dst_low = dst;
    this->dst_high = dst >> 32;
  }
};

struct WEdge14 {
  uint16_t src_low;
  uint16_t src_mid;
  uint16_t src_high;
  uint16_t dst_low;
  uint16_t dst_mid;
  uint16_t dst_high;
  uint8_t weight;
  uint8_t get_weight() const { return weight; }
  void set_weight(uint8_t weight) { this->weight = weight; }
  uint64_t get_src() const {
    uint64_t src = src_high;
    src <<= 16;
    src |= src_mid;
    src <<= 16;
    src |= src_low;
    return src;
  }
  uint64_t get_dst() const {
    uint64_t dst = dst_high;
    dst <<= 16;
    dst |= dst_mid;
    dst <<= 16;
    dst |= dst_low;
    return dst;
  }

  void set_src(uint64_t src) {
    this->src_low = src;
    this->src_mid = src >> 16;
    this->src_high = src >> 32;
  }
  void set_dst(uint64_t dst) {
    this->dst_low = dst;
    this->dst_mid = dst >> 16;
    this->dst_high = dst >> 32;
  }
};

struct WEdgeId16 {
  uint32_t src;
  uint32_t dst;
  uint64_t compound; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint8_t get_weight() const { return compound >> 56; }
  void set_weight(uint8_t w) {
    uint64_t tmp = w;
    compound |= tmp << 56;
  }
  void set_edge_id(uint64_t id) { compound = set_compound(id, get_weight()); }
  auto get_edge_id() const { return get_edge_id_external(compound); }

  uint64_t get_src() const { return src; }
  uint64_t get_dst() const { return dst; }
  void set_src(uint64_t src) { this->src = src; }
  void set_dst(uint64_t dst) { this->dst = dst; }
};

struct WEdgeId20 {
  uint32_t src_low;
  uint32_t dst_low;
  uint32_t compound_low; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint32_t compound_high;
  uint16_t src_high;
  uint16_t dst_high;
  uint8_t get_weight() const { return compound_high >> 24; }
  void set_weight(uint8_t w) {
    uint32_t tmp = w;
    compound_high |= tmp << 24;
  }
  void set_edge_id(uint64_t id) {
    uint64_t compound = set_compound(id, get_weight());
    compound_low = compound;
    compound_high = compound >> 32;
  }
  auto get_edge_id() const {
    uint64_t compound = compound_high;
    compound <<= 32;
    compound |= compound_low;
    return get_edge_id_external(compound);
  }
  uint64_t get_src() const {
    uint64_t src = src_high;
    src <<= 32;
    src |= src_low;
    return src;
  }
  uint64_t get_dst() const {
    uint64_t dst = dst_high;
    dst <<= 32;
    dst |= dst_low;
    return dst;
  }

  void set_src(uint64_t src) {
    this->src_low = src;
    this->src_high = src >> 32;
  }
  void set_dst(uint64_t dst) {
    this->dst_low = dst;
    this->dst_high = dst >> 32;
  }
};

struct WEdgeId24 {
  uint64_t src;
  uint64_t dst;
  uint64_t compound; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint8_t get_weight() const { return compound >> 56; }
  void set_weight(uint8_t w) {
    uint64_t tmp = w;
    compound |= tmp << 56;
  }
  void set_edge_id(uint64_t id) { compound = set_compound(id, get_weight()); }
  auto get_edge_id() const {
    return get_edge_id_external(compound);
  }
  uint64_t get_src() const { return src; }
  uint64_t get_dst() const { return dst; }
  void set_src(uint64_t src) { this->src = src; }
  void set_dst(uint64_t dst) { this->dst = src; }
};

inline std::ostream& operator<<(std::ostream& out, const WEdge10 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdge12 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdge14 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId16 edge) {
  return print_wedgeid(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId20 edge) {
  return print_wedgeid(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId24 edge) {
  return print_wedgeid(out, edge);
}
} // namespace hybridMST
