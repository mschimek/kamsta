#pragma once
#include <cstdint>
#include <ostream>
#include <utility>

namespace hybridMST {
/// Both values are stored in a 64 bit integer, weight starts at bit 64 -
/// bit_left_shift
template <std::uint8_t bit_left_shift = 56>
inline uint64_t set_compound(uint64_t id, uint64_t weight) {
  weight <<= bit_left_shift;
  weight |= id;
  return weight;
}

inline uint64_t get_edge_id_external(uint64_t compound) {
  compound &= 0x00FFFFFFFFFFFFFF;
  return compound;
}

template <typename WEdgeIdType>
std::ostream& print_wedgeid(std::ostream& out, const WEdgeIdType& e) {
  return out << "(" << e.get_src() << ", " << e.get_dst() << ", "
             << static_cast<int>(e.get_weight()) << ", " << e.get_edge_id()
             << ")";
}

template <typename WEdgeIdType>
std::ostream& print_wedge(std::ostream& out, const WEdgeIdType& e) {
  return out << "(" << e.get_src() << ", " << e.get_dst() << ", "
             << static_cast<int>(e.get_weight()) << ")";
}

struct WEdge_4_1 {
  uint16_t src_low;
  uint16_t src_high;
  uint16_t dst_low;
  uint16_t dst_high;
  uint8_t weight_;
  uint8_t get_weight() const { return weight_; }
  void set_weight(uint8_t weight) { weight_ = weight; }
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

struct WEdge_5_1 {
  uint32_t src_low;
  uint32_t dst_low;
  uint8_t src_high;
  uint8_t dst_high;
  uint8_t weight_;
  uint8_t get_weight() const { return weight_; }
  void set_weight(uint8_t weight) { weight_ = weight; }
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

struct WEdge_6_1 {
  uint16_t src_low;
  uint16_t src_mid;
  uint16_t src_high;
  uint16_t dst_low;
  uint16_t dst_mid;
  uint16_t dst_high;
  uint8_t weight_;
  uint8_t get_weight() const { return weight_; }
  void set_weight(uint8_t weight) { weight_ = weight; }
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

struct WEdge_4_4 {
  uint32_t src_;
  uint32_t dst_;
  uint32_t weight_;
  uint32_t get_weight() const { return weight_; }
  void set_weight(uint32_t weight) { weight_ = weight; }
  uint64_t get_src() const { return src_; }
  uint64_t get_dst() const { return dst_; }

  void set_src(uint64_t src) { src_ = src; }
  void set_dst(uint64_t dst) { dst_ = dst; }
};

struct WEdgeId_4_1_7 {
  uint32_t src_;
  uint32_t dst_;
  uint64_t compound; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint8_t get_weight() const { return compound >> 56; }
  void set_weight_and_edge_id(uint8_t weight, uint64_t id) {
    compound = set_compound(id, weight);
  }
  auto get_edge_id() const { return get_edge_id_external(compound); }

  uint64_t get_src() const { return src_; }
  uint64_t get_dst() const { return dst_; }
  void set_src(uint64_t src) { src_ = src; }
  void set_dst(uint64_t dst) { dst_ = dst; }
};

struct WEdgeId_6_1_7 {
  uint32_t src_low;
  uint32_t dst_low;
  uint32_t compound_low; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint32_t compound_high;
  uint16_t src_high;
  uint16_t dst_high;
  uint8_t get_weight() const { return compound_high >> 24; }
  void set_weight_and_edge_id(uint8_t weight, uint64_t id) {
    uint64_t compound = set_compound(id, weight);
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
  uint64_t src_;
  uint64_t dst_;
  uint64_t compound; // 63 .. 56 weight | 55 .. 0 edge_ids
  uint8_t get_weight() const { return compound >> 56; }
  void set_weight_and_edge_id(uint8_t weight, uint64_t id) {
    compound = set_compound(id, weight);
  }
  auto get_edge_id() const { return get_edge_id_external(compound); }
  uint64_t get_src() const { return src_; }
  uint64_t get_dst() const { return dst_; }
  void set_src(uint64_t src) { src_ = src; }
  void set_dst(uint64_t dst) { dst_ = dst; }
};

struct WEdgeId_4_4_8 {
  uint32_t src_;
  uint32_t dst_;
  uint32_t weight_;
  uint32_t edge_id_low;
  uint32_t edge_id_high;
  uint32_t get_weight() const { return weight_; }
  void set_weight_and_edge_id(uint32_t weight, uint64_t id) {
    weight_ = weight;
    edge_id_low = id;
    edge_id_high = id >> 32;
  }
  auto get_edge_id() const {
    std::uint64_t id = edge_id_high;
    id <<= 32;
    id |= edge_id_low;
    return id;
  }
  uint64_t get_src() const { return src_; }
  uint64_t get_dst() const { return dst_; }
  void set_src(uint64_t src) { src_ = src; }
  void set_dst(uint64_t dst) { dst_ = dst; }
};

inline std::ostream& operator<<(std::ostream& out, const WEdge_4_1 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdge_5_1 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdge_6_1 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdge_4_4 edge) {
  return print_wedge(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId_4_1_7 edge) {
  return print_wedgeid(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId_6_1_7 edge) {
  return print_wedgeid(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId24 edge) {
  return print_wedgeid(out, edge);
}
inline std::ostream& operator<<(std::ostream& out, const WEdgeId_4_4_8 edge) {
  return print_wedgeid(out, edge);
}
} // namespace hybridMST
