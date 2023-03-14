#pragma once

#include <algorithm>

#include <ips4o/ips4o.hpp>
#include <parlay/hash_table.h>
#include <util/macros.hpp>

#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "mpi/allgather.hpp"
#include "util/utils.hpp"

namespace hybridMST {
class VertexLocator {
  struct MaxEdgePe {
    Edge edge{VID_UNDEFINED, VID_UNDEFINED};
    PEID pe;
  };

public:
  VertexLocator() = default;
  VertexLocator(const Edge max_edge) {
    MaxEdgePe max_edge_pe{max_edge, ctx.rank()};
    max_edges = mpi::allgather(max_edge_pe);
    const auto it = std::remove_if(
        max_edges.begin(), max_edges.end(),
        [&](const auto& elem) { return !is_defined(elem.edge.get_src()); });
    max_edges.erase(it, max_edges.end());
  }
  // get owner of edge (src, dst, ...)
  PEID get_pe(const VId& src, const VId& dst) const {
    Edge edge_to_find{src, dst}; // TODO fix with std::tie-ish or something
    auto find_it =
        std::lower_bound(max_edges.begin(), max_edges.end(), edge_to_find,
                         [&](const MaxEdgePe& max_edge, const Edge& edge) {
                           return max_edge.edge < edge;
                         });
    return find_it != max_edges.end() ? find_it->pe : -1;
  }

private:
  mpi::MPIContext ctx;
  std::vector<MaxEdgePe> max_edges;
};

struct VertexLocator_Split {
  struct EdgeInterval {
    Edge min_edge_;
    Edge max_edge_;
    EdgeInterval() = default;
    EdgeInterval(Edge min_edge, Edge max_edge)
        : min_edge_(min_edge), max_edge_(max_edge) {}
    bool is_empty() const { return !is_defined(min_edge_.get_src()); }
    friend std::ostream& operator<<(std::ostream& out,
                                    const EdgeInterval& interval) {
      return out << "[" << interval.min_edge_ << " - " << interval.max_edge_
                 << "]";
    }
  };
  static EdgeInterval inf_sentinel() {
    const auto max_edge = Edge{VID_UNDEFINED, VID_UNDEFINED};
    return EdgeInterval(max_edge, max_edge);
  }
  VertexLocator_Split() = default;
  VertexLocator_Split(Edge min_edge, Edge max_edge)
      : ctx{}, v_min{min_edge.get_src()}, v_max{max_edge.get_src()} {
    EdgeInterval local_interval{min_edge, max_edge};
    org_edge_limits_ = mpi::allgather(local_interval);
    for (int i = 0; static_cast<std::size_t>(i) < org_edge_limits_.size();
         ++i) {
      if (!org_edge_limits_[i].is_empty())
        edge_limits_.emplace_back(org_edge_limits_[i], i);
    }
    // auto comparator = [](const EdgeInterval& lhs, const EdgeInterval& rhs) {

    // };
    // classifier = Classifier(edge_limits_, max_edge,

    min_pe_v_min = !is_defined(v_min) ? ctx.rank() : get_min_pe(v_min);
    min_pe_v_max = !is_defined(v_max) ? ctx.rank() : get_min_pe(v_max);
    max_pe_v_min = !is_defined(v_min) ? ctx.rank() : get_max_pe(v_min);
    max_pe_v_max = !is_defined(v_max) ? ctx.rank() : get_max_pe(v_max);
    is_home_of_v_min = min_pe_v_min == ctx.rank();
    is_home_of_v_max = min_pe_v_max == ctx.rank();
    is_v_min_split = min_pe_v_min != max_pe_v_min;
    is_v_max_split = min_pe_v_max != max_pe_v_max;
  }
  bool has_edges() const { return v_min != VID_UNDEFINED; }
  std::size_t get_n() const { return has_edges() ? v_max - v_min + 1 : 0ull; }
  VId local_id(VId global_id) const { return global_id - v_min; }
  bool is_local(VId v) const { return v_min <= v && v <= v_max; }
  VId get_v_max(PEID pe) const {
    return org_edge_limits_[pe].max_edge_.get_src();
  }
  VId get_v_min(PEID pe) const {
    return org_edge_limits_[pe].min_edge_.get_src();
  }
  bool is_vertex_split(VId v) const { return get_min_pe(v) != get_max_pe(v); }

  PEID get_min_pe(const Edge& edge) const {
    const auto it =
        std::lower_bound(edge_limits_.begin(), edge_limits_.end(), edge,
                         [](const std::pair<EdgeInterval, int>& interval,
                            const Edge& edge_internal) {
                           return interval.first.max_edge_ < edge_internal;
                         });
    // MPI_ASSERT_(it != edge_limits_.end(), "");
    const PEID pe = it->second;
    return pe;
  }

  PEID get_min_pe_or_sentinel(const Edge& edge, PEID sentinel = -1) const {
    const auto it =
        std::lower_bound(edge_limits_.begin(), edge_limits_.end(), edge,
                         [](const std::pair<EdgeInterval, int>& interval,
                            const Edge& comp_arg_edge) {
                           return interval.first.max_edge_ < comp_arg_edge;
                         });
    const PEID pe = it != edge_limits_.end() ? it->second : sentinel;
    return pe;
  }

  PEID get_min_pe(const VId& v) const { return get_min_pe(Edge{v, 0}); }

  PEID get_max_pe(const Edge& edge) const {
    const auto it =
        std::upper_bound(edge_limits_.begin(), edge_limits_.end(), edge,
                         [](const Edge& edge_comp_arg,
                            const std::pair<EdgeInterval, int>& interval) {
                           return edge_comp_arg < interval.first.min_edge_;
                         });
    const auto prev_it = std::prev(it, 1);
    return prev_it->second;
  }

  PEID get_max_pe(const VId& v) const { return get_max_pe(Edge{v, VID_MAX}); }

  mpi::MPIContext ctx;
  PEID min_pe_v_min;
  PEID min_pe_v_max;
  PEID max_pe_v_min;
  PEID max_pe_v_max;
  bool is_v_min_split = false;
  bool is_v_max_split = false;
  bool is_home_of_v_min = true;
  bool is_home_of_v_max = true;
  std::vector<std::pair<EdgeInterval, int>> edge_limits_;
  std::vector<EdgeInterval> org_edge_limits_;
  VId v_min;
  VId v_max;
  // Classifier classifier;
  friend std::ostream& operator<<(std::ostream& out,
                                  const VertexLocator_Split& infos) {
    out << "v_min: " << infos.v_min << " v_max: " << infos.v_max
        << " min_v_min " << infos.min_pe_v_min << " max_v_min "
        << infos.max_pe_v_min << " min_v_max " << infos.min_pe_v_max
        << " max_v_max " << infos.max_pe_v_max << " is_home_of_v_min "
        << infos.is_home_of_v_min << " is_home_of_v_max "
        << infos.is_home_of_v_max << " is_v_min_split: " << infos.is_v_min_split
        << " is_v_max_split: " << infos.is_v_max_split << "\n";
    for (auto intervals : infos.edge_limits_)
      out << intervals.first << " " << intervals.second << " ";
    return out;
  }
};

template <typename Container>
inline std::pair<VId, VId> find_min_max(const execution::sequential& /*policy*/,
                                        const Container& elems) {
  VId min_idx = VID_UNDEFINED;
  VId max_idx = VID_UNDEFINED;
  if (elems.empty())
    return std::make_pair(min_idx, max_idx);
  min_idx = max_idx = 0;
  for (std::size_t i = 0; i < elems.size(); ++i) {
    if (!(elems[min_idx] < elems[i]))
      min_idx = i;
    if (elems[max_idx] < elems[i])
      max_idx = i;
  }
  return std::make_pair(min_idx, max_idx);
}

template <typename Container>
inline std::pair<VId, VId> find_min_max(const execution::parallel& /*policy*/,
                                        const Container& edges) {
  using EdgeType = typename Container::value_type;
  VId min_idx = VID_UNDEFINED;
  VId max_idx = VID_UNDEFINED;
  if (edges.empty())
    return std::make_pair(min_idx, max_idx);
  min_idx = max_idx = 0;
  auto comp = SrcDstWeightOrder<EdgeType>{};
  auto [min_it, max_it] = parlay::minmax_element(
      edges,
      [&](const EdgeType& lhs, const EdgeType& rhs) { return comp(lhs, rhs); });
  min_idx = std::distance(edges.begin(), min_it);
  max_idx = std::distance(edges.begin(), max_it);
  return std::make_pair(min_idx, max_idx);
}

class EdgeIdDistribution {
public:
  struct PeLocalId {
    PEID pe;
    LocalEdgeId local_id;
  };

  static constexpr GlobalEdgeId get_id(const LocalEdgeId& localId,
                                       const PEID& rank) {
    return combine(static_cast<GlobalEdgeId>(rank),
                   static_cast<GlobalEdgeId>(localId));
  }
  static PeLocalId get_pe_localId(const GlobalEdgeId& id) {
    PeLocalId res;
    res.local_id = static_cast<LocalEdgeId>(id & k_lsb_set);
    res.pe = static_cast<PEID>(id >> k);
    return res;
  }
  constexpr static PEID get_pe(const GlobalEdgeId& id) {
    return static_cast<PEID>(id >> k);
  }
  constexpr static LocalEdgeId get_local_id(const GlobalEdgeId& id) {
    return static_cast<LocalEdgeId>(id & k_lsb_set);
  }

private:
  static constexpr GlobalEdgeId combine(GlobalEdgeId a, GlobalEdgeId b) {
    return (b & k_lsb_set) | (a << k);
  }

  static constexpr GlobalEdgeId zero = 0ull;
  static constexpr GlobalEdgeId k = 40ull;
  static constexpr GlobalEdgeId nb_shift_right = 64ull - k;
  static constexpr GlobalEdgeId k_lsb_set = (~zero) >> nb_shift_right;
};

struct CompactifyVerticesTrivial {
  template <typename... Args>
  static std::unordered_map<VId, VId>
  execute(const std::vector<Args...>& edges,
          non_init_vector<VId>& local_id_to_global_ids) {
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        edges.size(), parlay::hash_numeric<VId>{});

    parallel_for(0, edges.size(), [&](std::size_t i) {
      const auto& elem = edges[i];
      table.insert(elem.get_src());
    });
    auto local_vertices = table.entries();
    ips4o::parallel::sort(local_vertices.begin(), local_vertices.end());
    std::unordered_map<VId, VId> map;
    std::size_t counter = 0;
    VId prev = VID_UNDEFINED;
    for (const auto& v : local_vertices) {
      auto [it, is_inserted] = map.emplace(v, counter);
      counter += is_inserted;
    }
    paralle_for(0, local_vertices.size(), [&](std::size_t i) {
      auto local_vertex = local_vertices[i];
      auto it = map.find(local_vertex);
      local_id_to_global_ids[it->second] = it->first;
    });
    return map;
  }
};

struct LocalVertexCompactification {
  template <typename T, typename... Args>
  static auto get_max_num_local_vertices(const std::vector<T, Args...>& edges) {
    if (edges.empty()) {
      return 0ull;
    }
    const auto [min_it, max_it] = parlay::minmax_element(edges, SrcOrder<T>{});
    return (max_it->get_src() - min_it->get_src()) + 1ull;
  }
  template <typename T, typename... Args>
  static auto execute(const std::vector<T, Args...>& edges, int round) {

    get_timer().start("graph_internal_local_vertex_compact", round);
    // const auto max_num_local_vertices = get_max_num_local_vertices(edges);
    // TODO check why the above approach leads to very high running times
    // although place should be sufficient
    const std::size_t table_size =
        2 * (edges.size() +
             10); // std::max(10000ull, (max_num_local_vertices + 100ul) * 2);
    parlay::hashtable<parlay::hash_numeric<VId>> table(
        table_size, parlay::hash_numeric<VId>{});

    parallel_for(0, edges.size(), [&](std::size_t i) {
      const auto& elem = edges[i];
      table.insert(elem.get_src());
    });
    auto local_vertices = table.entries();
    if (local_vertices.size() <= 1) {
      get_timer().stop("graph_internal_local_vertex_compact", round);
      return local_vertices;
    }
    auto min_it = parlay::min_element(local_vertices);
    std::swap(local_vertices.front(), *min_it);
    auto max_it = parlay::max_element(local_vertices);
    std::swap(local_vertices.back(), *max_it);

    //{
    //  auto [min_it_, max_it_] = std::minmax_element(
    //      std::execution::par, local_vertices.begin(), local_vertices.end());
    //  MPI_ASSERT_(min_it_ == local_vertices.begin(), "");
    //  MPI_ASSERT_(max_it_ + 1 == local_vertices.end(), "");
    //}
    get_timer().stop("graph_internal_local_vertex_compact", round);
    return local_vertices;
  }
};

struct CompactifyVerticesParallel {
  template <typename CompVertices, typename Map>
  static auto execute(const CompVertices& compact_vertices, Map& map,
                      non_init_vector<VId>& local_id_to_global_ids, int round) {

    get_timer().start("graph_internal_local_vertex_compact_mapping", round);
    parallel_for(0, compact_vertices.size(), [&](std::size_t i) {
      const auto& v = compact_vertices[i];
      local_id_to_global_ids[i] = v;
      auto [it, is_inserted] = map.insert(v + 1, i);
      (void)it;
      // if (it == map.end()) {
      //   std::cout << "wrong growt insertion" << std::endl;
      //   std::abort();
      // }
    });
    get_timer().stop("graph_internal_local_vertex_compact_mapping", round);
  }
};
} // namespace hybridMST
