#pragma once

#include <new>

#include <numeric>
#include <tuple>
#include <util/macros.hpp>
#include <utility>

#include "parlay/hash_table.h"

#include "definitions.hpp"
#include "mpi/context.hpp"
#include "shared_mem_parallel.hpp"
#include "util/timer.hpp"

namespace hybridMST {

template <typename T> class Span {
public:
  using value_type = T;
  Span() : Span(nullptr, 0) {}
  Span(T* ptr, std::size_t size)
      : ptr_(ptr), size_(size), initial_size_(size) {}
  template <typename... Args>
  Span(std::vector<T, Args...>& vector)
      : ptr_(std::data(vector)), size_(vector.size()) {}
  template <typename... Args>
  Span(const std::vector<Args...>& vector)
      : ptr_(std::data(vector)), size_(vector.size()) {}
  T* data() { return ptr_; };
  T* data() const { return ptr_; };
  std::size_t size() const { return size_; }
  std::size_t initial_size() const { return initial_size_; }
  T& operator[](std::size_t i) { return ptr_[i]; }
  const T& operator[](std::size_t i) const { return ptr_[i]; }
  void resize(std::size_t new_size) { size_ = new_size; }
  const T* begin() const { return ptr_; }
  T* begin() { return ptr_; }
  const T* end() const { return ptr_ + size_; }
  T* end() { return ptr_ + size_; }
  bool empty() const { return size_ == 0; }

private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0ull;
  std::size_t initial_size_ = 0ull;
};
template <typename T, typename... Args>
Span(const std::vector<T, Args...>& vector) -> Span<const T>;

template <typename Edges>
inline std::uint64_t sum_edge_weights(const Edges& edges) {
  std::uint64_t sum = 0;
  for (const auto& edge : edges) {
    sum += edge.get_weight();
  }
  return sum;
}
struct False_Predicate {
  template <typename... Args>
  constexpr bool operator()(Args&&...) const noexcept {
    return false;
  }
};

struct Identity {
  template <typename T> constexpr T&& operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }
  template <typename T>
  constexpr T&& operator()(T&& t, const std::size_t&) const noexcept {
    return std::forward<T>(t);
  }
};
template <typename EdgeType> struct WeightOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return lhs.get_weight() < rhs.get_weight();
  }
};
template <typename EdgeType> struct WeightSrcDstOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return std::make_tuple(lhs.get_weight(), lhs.get_src(), lhs.get_dst()) <
           std::make_tuple(rhs.get_weight(), rhs.get_src(), rhs.get_dst());
  }
};
template <typename EdgeType> struct SrcOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return lhs.get_src() < rhs.get_src();
  }
};
template <typename EdgeType> struct DstOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return lhs.get_dst() < rhs.get_dst();
  }
};

template <typename EdgeType> struct DstSrcWeightOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return std::make_tuple(lhs.get_dst(), lhs.get_src(), lhs.get_weight()) <
           std::make_tuple(rhs.get_dst(), rhs.get_src(), rhs.get_weight());
  }
};

template <typename EdgeType> struct SrcWeightOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return std::make_tuple(lhs.get_src(), lhs.get_weight()) <
           std::make_tuple(rhs.get_src(), rhs.get_weight());
  }
};

template <typename EdgeType> struct SrcDstOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return std::make_tuple(lhs.get_src(), lhs.get_dst()) <
           std::make_tuple(rhs.get_src(), rhs.get_dst());
  }
};

template <typename EdgeType> struct SrcDstWeightOrder {
  bool operator()(const EdgeType& lhs, const EdgeType& rhs) const {
    return std::make_tuple(lhs.get_src(), lhs.get_dst(), lhs.get_weight()) <
           std::make_tuple(rhs.get_src(), rhs.get_dst(), rhs.get_weight());
  }
};

template <typename EdgeTypeLhs, typename EdgeTypeRhs> struct SrcDstWeightEqual {
  bool operator()(const EdgeTypeLhs& lhs, const EdgeTypeRhs& rhs) const {
    return std::make_tuple(lhs.get_src(), lhs.get_dst(), lhs.get_weight()) ==
           std::make_tuple(rhs.get_src(), rhs.get_dst(), rhs.get_weight());
  }
};

struct SrcDstWeightIdOrder {
  bool operator()(const WEdgeId& lhs, const WEdgeId& rhs) const {
    return std::make_tuple(lhs.get_src(), lhs.get_dst(), lhs.get_weight(),
                           lhs.get_global_id()) <
           std::make_tuple(rhs.get_src(), rhs.get_dst(), rhs.get_weight(),
                           rhs.get_global_id());
  }
};

template <typename Container, typename F>
void map(Container& container, F&& f) {
  parallel_for(0, container.size(), [&](std::size_t i) { f(container[i], i); });
}

template <typename Container, typename Transform>
auto filter_out_duplicates(const Container& container, Transform&& transform) {
  using ValueType = typename Container::value_type;
  static_assert(std::is_integral_v<ValueType>);
  parlay::hashtable<parlay::hash_numeric<ValueType>> table(
      container.size(), parlay::hash_numeric<ValueType>{});
  parallel_for(0, container.size(),
               [&](std::size_t i) { table.insert(transform(container[i])); });
  return table.entries();
}

template <typename It, typename F> void map(It begin, It end, F&& f) {
  const std::size_t size = std::distance(begin, end);
  parallel_for(0, size, [&](std::size_t i) { f(begin[i], i); });
}

template <typename Container, typename Init>
void assign_initialize(Container& container, Init&& init) {
  parallel_for(0, container.size(),
               [&](std::size_t i) { container[i] = init(i); });
}

template <typename It, typename Init>
void assign_initialize(It begin, It end, Init&& init) {
  const std::size_t size = std::distance(begin, end);
  parallel_for(0, size, [&](std::size_t i) { begin[i] = init(i); });
}

template <typename T, typename Container>
void append_second_to_first(non_init_vector<T>& vec,
                            const Container& data_to_append) {
  const std::size_t num_elems = vec.size();
  const std::size_t num_elems_to_append = data_to_append.size();
  vec.resize(num_elems + num_elems_to_append);
  parallel_for(0, num_elems_to_append,
               [&](std::size_t i) { vec[num_elems + i] = data_to_append[i]; });
}

template <typename Container1, typename Container2>
non_init_vector<typename Container1::value_type>
combine(const Container1& container1, const Container2& container2) {
  using T1 = typename std::remove_cv_t<Container1>::value_type;
  using T2 = typename std::remove_cv_t<Container2>::value_type;
  static_assert(std::is_same_v<T1, T2>);
  non_init_vector<T1> recv_container(container1.size() + container2.size());
  const std::size_t size = container1.size();
  parallel_for(0, container1.size(),
               [&](std::size_t i) { recv_container[i] = container1[i]; });

  parallel_for(0, container2.size(), [&](std::size_t i) {
    recv_container[size + i] = container2[i];
  });
  return recv_container;
}

inline bool is_local(const VId v, const VertexRange_& range) {
  return range.v_begin <= v && v < range.v_end;
}

template <typename Edge>
bool is_local(const Edge& edge, const VertexRange_& range) {
  return is_local(edge.get_src(), range) && is_local(edge.get_dst(), range);
}

template <typename Container> bool are_elements_unique(Container container) {
  std::sort(container.begin(), container.end());
  auto it = std::adjacent_find(container.begin(), container.end());
  return it == container.end();
}

template <typename Container, typename GetKey>
bool are_keys_unique(const Container& container, GetKey&& get_key) {
  using Key = std::invoke_result_t<GetKey, typename Container::value_type>;
  non_init_vector<Key> keys(container.size());
  std::transform(container.begin(), container.end(), keys.begin(),
                 [&](const auto& elem) { return get_key(elem); });
  std::sort(keys.begin(), keys.end());
  auto it = std::adjacent_find(keys.begin(), keys.end());
  return it == keys.end();
}

template <typename T>
struct alignas(128) // std::hardware_constructive_interference_size is not
                    // known? TODO check on this
    CachelineAlignedType {
  CachelineAlignedType(T&& t) : value(t) {
    static_assert(std::is_integral_v<T>);
  }
  operator T&() { return value; }
  T value;
};

template <typename T> T set_MSB(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr VId ones = ~zero;
  constexpr VId all_ones_but_MSB = ones >> 1;
  constexpr VId only_MSB_set = ~all_ones_but_MSB;
  return t | only_MSB_set;
}

template <typename T> T reset_MSB(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr T ones = ~zero;
  constexpr T all_ones_but_MSB = ones >> 1;
  return t & all_ones_but_MSB;
}

template <typename T> bool is_MSB_set(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr VId ones = ~zero;
  constexpr VId all_ones_but_MSB = ones >> 1;
  constexpr VId only_MSB_set = ~all_ones_but_MSB;
  return t & only_MSB_set;
}

template <typename T> T set_TMSB(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr VId ones = ~zero;
  constexpr VId all_ones_but_TMSB = ones >> 2;
  constexpr VId only_TMSB_set = ~all_ones_but_TMSB;
  return t | only_TMSB_set;
}

template <typename T> T reset_TMSB(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr T ones = ~zero;
  constexpr T all_ones_but_TMSB = ones >> 2;
  return t & all_ones_but_TMSB;
}

template <typename T> bool is_TMSB_set(T t) {
  static_assert(std::is_integral_v<T>);
  static_assert(std::is_unsigned_v<T>);
  constexpr T zero = 0u;
  constexpr VId ones = ~zero;
  constexpr VId all_ones_but_TMSB = ones >> 2;
  constexpr VId only_TMSB_set = ~all_ones_but_TMSB;
  return (t & only_TMSB_set) == only_TMSB_set;
}

std::pair<std::vector<WEdge>, VertexRange> inline convert(
    const std::pair<std::vector<graphs::WEdge>, VertexRange>& p) {
  auto res = std::make_pair(std::vector<WEdge>{}, p.second);
  res.first.resize(p.first.size());
  for (std::size_t i = 0; i < p.first.size(); ++i) {
    auto edge = p.first[i];
    res.first[i].set_src(edge.get_src());
    res.first[i].set_dst(edge.get_dst());
    res.first[i].set_weight(edge.get_weight());
  }
  return res;
}

template <typename Edge> Edge flip_edge(const Edge& edge) {
  Edge flipped_edge = edge;
  flipped_edge.set_src(edge.get_dst());
  flipped_edge.set_dst(edge.get_src());
  return flipped_edge;
}

template <typename Edges> Edges inline flip_edges(const Edges& edges) {
  Edges flipped_edges(edges.size());
  parallel_for(0, edges.size(), [&](std::size_t i) {
    auto& flipped_edge = flipped_edges[i];
    const auto& edge = edges[i];
    flipped_edge = flip_edge(edge);
  });
  return flipped_edges;
}
template <typename Container> void dump(Container& cont) {
  Container tmp;
  std::swap(tmp, cont);
}

// for debugging purposes only
inline void wait_for_user(const std::string& desc) {
  mpi::MPIContext ctx;
  REORDERING_BARRIER;
  if (ctx.rank() == 0) {
    int i = 0;
    std::cout << desc << std::endl;
    std::cin >> i;
    [[maybe_unused]] int volatile ii = i;
  }
  REORDERING_BARRIER;
}

struct sum {
  std::uint64_t operator()(const std::uint64_t& lhs,
                           const std::uint64_t& rhs) const {
    return lhs + rhs;
  }
};
struct max {
  std::uint64_t operator()(const std::uint64_t& lhs,
                           const std::uint64_t& rhs) const {
    return std::max(lhs, rhs);
  }
};
template <typename Edges> inline void print_statistics(const Edges& edges) {
  if (edges.empty()) {
    return;
  }
  const std::size_t local_n = edges.back().get_src() + 1;
  const std::size_t n = mpi::allreduce_max(local_n);
  const std::size_t log_n = std::log2(n);
  std::vector<std::size_t> num_buckets(log_n + 2, 0);
  std::size_t cur_degree = 0;
  auto prev_src = edges.front().get_src();
  for (const auto& edge : edges) {
    if (edge.get_src() == prev_src) {
      ++cur_degree;
    } else {
      ++num_buckets[std::log2(cur_degree)];
      cur_degree = 1;
      prev_src = edge.get_src();
    }
  }
  mpi::MPIContext ctx;
  const auto sums = mpi::allreduce(num_buckets, sum{});
  const auto maximas = mpi::allreduce(num_buckets, max{});
  if (ctx.rank() == 0) {
    for (std::size_t i = 0; i < num_buckets.size(); ++i) {
      std::size_t start = 1ull << i;
      std::size_t end = 1ull << (i + 1);
      std::cout << "[" << std::setw(10) << start << ", " << std::setw(10) << end
                << "): " << maximas[i] << " " << sums[i] << std::endl;
    }
  }
}

/// Compute number of local vertices assuming the array to be lexicographically
/// sorted
template <typename Edges>
std::size_t get_number_local_vertices(const Edges& edges) {
  if (edges.empty())
    return 0;
  return (edges.back().get_src() - edges.front().get_src()) + 1;
}

/// Returns max local vertice ID assuming the array to be lexicographically
/// sorted.
template <typename Edges> std::size_t get_max_local_vertex(const Edges& edges) {
  if (edges.empty())
    return 0;
  return edges.back().get_src();
}

inline void print_on_root(const std::string& msg,
                          const mpi::MPIContext& ctx = mpi::MPIContext()) {
  if (ctx.rank() == 0) {
    std::cout << msg << std::endl;
  }
}

template <typename EdgeType> std::string get_edge_type_name() {
  if (std::is_same_v<WEdge_4_1, EdgeType>) {
    return "WEdge_4_1";
  }
  if (std::is_same_v<WEdge_5_1, EdgeType>) {
    return "WEdge_5_1";
  }
  if (std::is_same_v<WEdge_6_1, EdgeType>) {
    return "WEdge_6_1";
  }
  if (std::is_same_v<WEdge_4_4, EdgeType>) {
    return "WEdge_4_4";
  }
  if (std::is_same_v<WEdge, EdgeType>) {
    return "WEdge_8_4";
  }
  return "unknown edge type";
}

template <typename Edges>
inline VertexRange_ create_vertex_range(const Edges& edges) {
  if (edges.empty()) {
    return VertexRange_{VID_UNDEFINED, VID_UNDEFINED};
  }
  return VertexRange_{edges.front().get_src(), edges.back().get_src() + 1};
}

///@brief For local edges (u,v) for which u and v can be represented with 32
/// bits this functions returns (u << 32 | v)
template <typename EdgeType>
static VId get_combined_id(EdgeType& local_edge, VId v_min) {
  VId combined_value = local_edge.get_src() - v_min;
  combined_value <<= 32;
  combined_value |= (local_edge.get_dst() - v_min);
  return combined_value;
}

} // namespace hybridMST
