#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "edge_types.hpp"
#include "graphs/interface.hpp"
#include "util/allocators.hpp"

namespace hybridMST {
template <typename T>
using non_init_vector = std::vector<T, default_init_allocator<T>>;
using concurrent_bool_vector =
    std::vector<std::uint8_t, default_init_allocator<std::uint8_t>>;

using VId = graphs::VId;
using LocalVId = std::uint32_t;
using LocalEdgeId = std::uint32_t;
using GlobalEdgeId = std::uint64_t;
using Weight = std::uint32_t;
using PEID = int;

constexpr VId VID_UNDEFINED = std::numeric_limits<VId>::max();
inline bool is_defined(const VId& v) { return v != VID_UNDEFINED; }
constexpr VId VID_MAX = std::numeric_limits<VId>::max() - 1;
constexpr Weight WEIGHT_UNDEFINED = std::numeric_limits<Weight>::max();
inline bool is_defined(const Weight& w) { return w != WEIGHT_UNDEFINED; }
constexpr Weight WEIGHT_INF = std::numeric_limits<Weight>::max() - 1;
constexpr Weight WEIGHT_MAX = std::numeric_limits<Weight>::max() - 2;
constexpr LocalEdgeId LOCAL_EDGEID_UNDEFINED =
    std::numeric_limits<LocalEdgeId>::max();
static_assert(WEIGHT_UNDEFINED == LOCAL_EDGEID_UNDEFINED);
// inline bool is_defined(const LocalEdgeId& id) { return id !=
// LOCAL_EDGEID_UNDEFINED; } same type as Weight (use strong typing?)
constexpr LocalEdgeId LOCAL_EDGEID_MAX =
    std::numeric_limits<LocalEdgeId>::max() - 1;
constexpr LocalVId LOCAL_VID_UNDEFINED = std::numeric_limits<LocalVId>::max();
constexpr LocalVId LOCAL_VID_MAX = std::numeric_limits<LocalVId>::max() - 1;
static_assert(WEIGHT_UNDEFINED == LOCAL_VID_UNDEFINED);
// inline bool is_defined(const LocalVId& id) { return id !=
// LOCAL_VID_UNDEFINED; } same type as Weight (use strong typing?)

constexpr GlobalEdgeId GLOBAL_EDGEID_UNDEFINED =
    std::numeric_limits<GlobalEdgeId>::max();
static_assert(WEIGHT_UNDEFINED == LOCAL_EDGEID_UNDEFINED);
// inline bool is_defined(const GlobalEdgeId& id) { return id !=
static_assert(GLOBAL_EDGEID_UNDEFINED == VID_UNDEFINED);
// GLOBAL_EDGEID_UNDEFINED; } same type as VId (use strong typing?)
constexpr GlobalEdgeId GLOBAL_EDGEID_MAX =
    std::numeric_limits<GlobalEdgeId>::max() - 1;

using VertexRange = graphs::VertexRange;

struct VertexRange_ {
  VertexRange_() = default;
  VertexRange_(VId _v_begin, VId _v_end) : v_begin{_v_begin}, v_end{_v_end} {}
  std::size_t n() const { return v_end - v_begin; }
  VId v_begin = VID_UNDEFINED;
  VId v_end = VID_UNDEFINED;
};

struct WEdge {
  VId src_;
  VId dst_;
  Weight weight_;
  WEdge() = default;
  WEdge(VId src, VId dst, Weight weight)
      : src_{src}, dst_{dst}, weight_{weight} {}
  Weight get_weight() const { return weight_; }
  void set_weight(Weight w) { weight_ = w; }
  VId get_src() const { return src_; }
  void set_src(VId src) { src_ = src; }
  VId get_dst() const { return dst_; }
  void set_dst(VId dst) { dst_ = dst; }

  friend std::ostream& operator<<(std::ostream& out, const WEdge& edge) {
    return out << "(" << edge.src_ << ", " << edge.dst_ << ", " << edge.weight_
               << ")";
  }
};

struct WEdgeId {
  VId src_;
  VId dst_;
  Weight weight_;
  GlobalEdgeId global_id_;
  Weight get_weight() const { return weight_; }
  VId get_src() const { return src_; }
  void set_src(VId src) { src_ = src; }
  VId get_dst() const { return dst_; }
  void set_dst(VId dst) { dst_ = dst; }
  GlobalEdgeId get_global_id() const { return global_id_; }
  void set_weight_and_edge_id(uint64_t weight, uint64_t id) {
    weight_ = weight;
    global_id_ = id;
  }
  std::uint64_t get_edge_id() const { return global_id_; }

  WEdgeId() = default;
  WEdgeId(VId src, VId dst, Weight weight, GlobalEdgeId global_id)
      : src_{src}, dst_{dst}, weight_{weight}, global_id_{global_id} {}
  WEdgeId(const WEdge& edge, GlobalEdgeId global_id)
      : src_{edge.get_src()}, dst_{edge.get_dst()}, weight_{edge.get_weight()},
        global_id_{global_id} {}

  friend std::ostream& operator<<(std::ostream& out, const WEdgeId& edge) {
    return out << "(" << edge.get_src() << ", " << edge.get_dst() << ", "
               << edge.get_weight() << ", " << edge.get_global_id() << ")";
  }
};

inline bool operator<(const WEdge& lhs, const WEdge& rhs) {
  return std::tie(lhs.src_, lhs.dst_, lhs.weight_) <
         std::tie(rhs.src_, rhs.dst_, rhs.weight_);
}
inline bool operator==(const WEdge& lhs, const WEdge& rhs) {
  return std::tie(lhs.src_, lhs.dst_, lhs.weight_) ==
         std::tie(rhs.src_, rhs.dst_, rhs.weight_);
}
inline bool operator<(const WEdgeId& lhs, const WEdgeId& rhs) {
  return std::tie(lhs.src_, lhs.dst_, lhs.weight_, lhs.global_id_) <
         std::tie(rhs.src_, rhs.dst_, rhs.weight_, rhs.global_id_);
}

template <typename EdgeType>
inline bool is_equal(const EdgeType& lhs, const EdgeType& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  return std::tie(lhs_src, lhs_dst, lhs_weight) ==
         std::tie(rhs_src, rhs_dst, rhs_weight);
}

inline bool operator==(const WEdge_4_1& lhs, const WEdge_4_1& rhs) {
  return is_equal(lhs, rhs);
}

inline bool operator==(const WEdge_5_1& lhs, const WEdge_5_1& rhs) {
  return is_equal(lhs, rhs);
}

inline bool operator==(const WEdge_6_1& lhs, const WEdge_6_1& rhs) {
  return is_equal(lhs, rhs);
}

inline bool operator==(const WEdge_4_4& lhs, const WEdge_4_4& rhs) {
  return is_equal(lhs, rhs);
}

inline bool operator<(const WEdgeId_4_1_7& lhs, const WEdgeId_4_1_7& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) <
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}
inline bool operator==(const WEdgeId_4_1_7& lhs, const WEdgeId_4_1_7& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) ==
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}
inline bool operator<(const WEdgeId_6_1_7& lhs, const WEdgeId_6_1_7& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) <
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}
inline bool operator==(const WEdgeId_6_1_7& lhs, const WEdgeId_6_1_7& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) ==
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}
inline bool operator<(const WEdgeId_4_4_8& lhs, const WEdgeId_4_4_8& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) <
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}
inline bool operator==(const WEdgeId_4_4_8& lhs, const WEdgeId_4_4_8& rhs) {
  const auto& lhs_src = lhs.get_src();
  const auto& lhs_dst = lhs.get_dst();
  const auto& lhs_weight = lhs.get_weight();
  const auto& lhs_id = lhs.get_edge_id();
  const auto& rhs_src = rhs.get_src();
  const auto& rhs_dst = rhs.get_dst();
  const auto& rhs_weight = rhs.get_weight();
  const auto& rhs_id = rhs.get_edge_id();
  return std::tie(lhs_src, lhs_dst, lhs_weight, lhs_id) ==
         std::tie(rhs_src, rhs_dst, rhs_weight, rhs_id);
}

inline bool operator==(const WEdgeId& lhs, const WEdgeId& rhs) {
  return std::tie(lhs.src_, lhs.dst_, lhs.weight_, lhs.global_id_) ==
         std::tie(rhs.src_, rhs.dst_, rhs.weight_, rhs.global_id_);
}

struct Edge {
  VId src_;
  VId dst_;
  VId get_src() const { return src_; }
  void set_src(VId src) { src_ = src; }
  VId get_dst() const { return dst_; }
  void set_dst(VId dst) { dst_ = dst; }
  Edge() = default;
  Edge(uint64_t src, uint64_t dst) : src_{src}, dst_{dst} {}
  friend std::ostream& operator<<(std::ostream& out, const Edge& edge) {
    return out << "(" << edge.get_src() << ", " << edge.get_dst() << ")";
  }
};

inline bool operator<(const Edge& lhs, const Edge& rhs) {
  return std::tie(lhs.src_, lhs.dst_) < std::tie(rhs.src_, rhs.dst_);
}
inline bool operator==(const Edge& lhs, const Edge& rhs) {
  return std::tie(lhs.src_, lhs.dst_) == std::tie(rhs.src_, rhs.dst_);
}
using WEdgeList = std::vector<WEdge>;
using WEdgeList14 = std::vector<WEdge_6_1>;
using EdgeList =
    std::vector<std::pair<long long unsigned int, long long unsigned int>>;
// template <typename EdgeType> inline VId src(const EdgeType& edge) {
//   return edge.src;
// }
// template <typename EdgeType> inline VId& src_ref(EdgeType& edge) {
//   return edge.src;
// }
// template <typename EdgeType> inline const VId& src_ref(const EdgeType& edge)
// {
//   return edge.src;
// }
// template <typename EdgeType> inline VId dst(const EdgeType& edge) {
//   return edge.dst;
// }
// template <typename EdgeType> inline VId& dst_ref(EdgeType& edge) {
//   return edge.dst;
// }
// template <typename EdgeType> inline const VId& dst_ref(const EdgeType& edge)
// {
//   return edge.dst;
// }
// template <typename EdgeType> Weight weight(const EdgeType& edge) {
//   return edge.weight;
// }
// template <typename EdgeType>
// inline const Weight& weight_ref(const EdgeType& edge) {
//   return edge.weight;
// }
// template <typename EdgeType> inline Weight& weight_ref(EdgeType& edge) {
//   return edge.weight;
// }

struct EdgeIdWeight {
  LocalEdgeId edge_id;
  Weight weight;
  friend std::ostream& operator<<(std::ostream& out, const EdgeIdWeight& elem) {
    return out << "(" << elem.edge_id << ", " << elem.weight << ")";
  }
};

struct LocalVertexWeight {
  LocalVId vertex;
  Weight weight;
  friend std::ostream& operator<<(std::ostream& out,
                                  const LocalVertexWeight& elem) {
    return out << "(" << elem.vertex << ", " << elem.weight << ")";
  }
};

struct EdgeIdWeightDst {
  LocalEdgeId edge_id;
  Weight weight;
  VId dst;
  friend std::ostream& operator<<(std::ostream& out,
                                  const EdgeIdWeightDst& elem) {
    return out << "(id: " << elem.edge_id << ", dst:" << elem.dst
               << ", w:" << elem.weight << ")";
  }
};

namespace execution {
struct sequential {};
struct parallel {};
} // namespace execution

} // namespace hybridMST
