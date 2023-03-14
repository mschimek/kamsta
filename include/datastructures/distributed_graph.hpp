#pragma once

#include <atomic>

#include <datastructures/sparse_distributed_graph_helpers.hpp>
#include <ips4o/ips4o.hpp>

#include "datastructures/distributed_graph_helper.hpp"
#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "util/atomic_ops.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"

namespace hybridMST {

/// the following two class specialization are (presumably) needed to different
/// at compile time between compactifying and non compactifying distributed
/// graphs.
template <bool compactify_internal>
struct DistributedGraph_CompactificationData {
  template <typename Edges>
  DistributedGraph_CompactificationData(const Edges& edges, std::size_t round)
      : compactified_vertices{LocalVertexCompactification::execute(edges,
                                                                   round)},
        global_id_to_local_id(compactified_vertices.size() * 1.25),
        local_id_to_global_id(compactified_vertices.size()),
        num_compactified_vertices{compactified_vertices.size()} {
    CompactifyVerticesParallel::execute(compactified_vertices,
                                        global_id_to_local_id,
                                        local_id_to_global_id, round);
  }

  parlay::sequence<VId> compactified_vertices;
  growt::GlobalVIdMap<VId> global_id_to_local_id;
  non_init_vector<VId> local_id_to_global_id;
  std::size_t num_compactified_vertices = 0;
};
template <> struct DistributedGraph_CompactificationData<false> {
  template <typename Edges>
  DistributedGraph_CompactificationData(const Edges& /*edges*/,
                                        std::size_t /*round*/) {}
};

class LocatorWrapper {
public:
  using PESplitStatus = sparse_graph::VertexLocator::PESplitStatus;
  constexpr static bool enable_debug = false;
  LocatorWrapper() = default;
  template <typename Edges>
  LocatorWrapper(Edge min_edge, Edge max_edge, const Edges& edges)
      : use_sparse_locator_{use_sparse_vertex_locator(edges.size())} {
    if (use_sparse_locator_) {
      non_init_vector<Edge> flipped_edges(edges.size());
      parallel_for(0, edges.size(), [&](std::size_t i) {
        flipped_edges[i] = Edge{edges[i].get_dst(), edges[i].get_src()};
      });
      sparse_vertex_locator.init(min_edge, max_edge, flipped_edges);
    } else {
      dense_vertex_locator = VertexLocator_Split(min_edge, max_edge);
    }
  }
  template <typename Res, typename Request>
  void check(Res res_sparse, Res res_dense, std::string operation,
             Request request) const {
    mpi::MPIContext ctx;
    if (res_sparse != res_dense) {
      std::cout << "rank: " << ctx.rank() << " " << operation
                << " request: " << request << " res_sparse: " << res_sparse
                << " res_dense: " << res_dense << std::endl;
      std::cout << sparse_vertex_locator.debug_print() << std::endl;
      std::cout << dense_vertex_locator << std::endl;
    }
  }
  [[nodiscard]] PEID get_min_pe(const Edge& edge) const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.get_min_pe(edge);
      return dense_vertex_locator.get_min_pe(edge);
    } else {
      auto res_sparse = sparse_vertex_locator.get_min_pe(edge);
      auto res_dense = dense_vertex_locator.get_min_pe(edge);
      check(res_sparse, res_dense, "get_min_edge", edge);
      return res_dense;
    }
  }
  [[nodiscard]] PEID get_min_pe_or_sentinel(const Edge& edge,
                                            PEID sentinel = -1) const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.get_min_pe_or_sentinel(edge, sentinel);
      return dense_vertex_locator.get_min_pe_or_sentinel(edge, sentinel);
    } else {
      auto res_sparse =
          sparse_vertex_locator.get_min_pe_or_sentinel(edge, sentinel);
      auto res_dense =
          dense_vertex_locator.get_min_pe_or_sentinel(edge, sentinel);
      check(res_sparse, res_dense, "get_min_edge", edge);
      return res_dense;
    }
  }
  [[nodiscard]] PESplitStatus get_min_pe_split_info(const Edge& edge) const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.get_min_pe_and_split_info(edge);
      const auto min_pe = dense_vertex_locator.get_min_pe(edge);
      const auto min_pe_ = dense_vertex_locator.get_min_pe(edge.get_src());
      const auto max_pe_ = dense_vertex_locator.get_max_pe(edge.get_src());
      return PESplitStatus{min_pe, min_pe_ != max_pe_};
    }
    {
      auto res_sparse = sparse_vertex_locator.get_min_pe_and_split_info(edge);
      const auto min_pe = dense_vertex_locator.get_min_pe(edge);
      const auto min_pe_ = dense_vertex_locator.get_min_pe(edge.get_src());
      const auto max_pe_ = dense_vertex_locator.get_max_pe(edge.get_src());
      auto res_dense = PESplitStatus{min_pe, min_pe_ != max_pe_};
      check(res_sparse, res_dense, "get_min_edge_pe_split", edge);
      return res_dense;
    }
  }
  [[nodiscard]] bool is_v_min_split() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.is_v_min_split();
      return dense_vertex_locator.is_v_min_split;
    } else {
      auto res_sparse = sparse_vertex_locator.is_v_min_split();
      auto res_dense = dense_vertex_locator.is_v_min_split;
      check(res_sparse, res_dense, "is_v_min_split", 0);
      return res_dense;
    }
  }
  [[nodiscard]] bool is_v_max_split() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.is_v_max_split();
      return dense_vertex_locator.is_v_max_split;
    } else {
      auto res_sparse = sparse_vertex_locator.is_v_max_split();
      auto res_dense = dense_vertex_locator.is_v_max_split;
      check(res_sparse, res_dense, "is_v_max_split", 0);
      return res_dense;
    }
  }
  [[nodiscard]] bool is_home_of_v_min() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.is_home_of_v_min();
      return dense_vertex_locator.is_home_of_v_min;
    } else {
      auto res_sparse = sparse_vertex_locator.is_home_of_v_min();
      auto res_dense = dense_vertex_locator.is_home_of_v_min;
      check(res_sparse, res_dense, "is_home_of_v_min", 0);
      return res_dense;
    }
  }
  [[nodiscard]] bool is_home_of_v_max() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.is_home_of_v_max();
      return dense_vertex_locator.is_home_of_v_max;
    } else {
      auto res_sparse = sparse_vertex_locator.is_home_of_v_max();
      auto res_dense = dense_vertex_locator.is_home_of_v_max;
      check(res_sparse, res_dense, "is_home_of_v_max", 0);
      return res_dense;
    }
  }
  [[nodiscard]] bool is_local(VId v) const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.is_local(v);
      return dense_vertex_locator.is_local(v);
    } else {
      auto res_sparse = sparse_vertex_locator.is_local(v);
      auto res_dense = dense_vertex_locator.is_local(v);
      check(res_sparse, res_dense, "is_local", v);
      return res_dense;
    }
  }
  [[nodiscard]] VId v_min() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.v_min();
      return dense_vertex_locator.v_min;
    } else {
      auto res_sparse = sparse_vertex_locator.v_min();
      auto res_dense = dense_vertex_locator.v_min;
      check(res_sparse, res_dense, "v_min", 0);
      return res_dense;
    }
  }
  [[nodiscard]] VId v_max() const {
    if constexpr (!enable_debug) {
      if (use_sparse_locator_)
        return sparse_vertex_locator.v_max();
      return dense_vertex_locator.v_max;
    } else {
      auto res_sparse = sparse_vertex_locator.v_max();
      auto res_dense = dense_vertex_locator.v_max;
      check(res_sparse, res_dense, "v_max", 0);
      return res_dense;
    }
  }

private:
  bool use_sparse_vertex_locator(std::size_t num_edges) {
    mpi::MPIContext ctx;
    return (ctx.size() * 2ull) >= mpi::allreduce_max(num_edges);
  }

  bool use_sparse_locator_ = false;
  VertexLocator_Split dense_vertex_locator;
  sparse_graph::VertexLocator sparse_vertex_locator;
};

template <typename EdgeType_, bool compactify = false> class DistributedGraph {
public:
  using EdgeType = EdgeType_;
  template <typename T> using container = non_init_vector<T>;
  DistributedGraph(non_init_vector<EdgeType_>& edges, std::size_t round)
      : compactification_data{edges, round}, _edges{edges} {
    get_timer().start("graph_internal_init_setup", round);
    REORDERING_BARRIER
    Edge unweighted_min_edge = Edge{VID_UNDEFINED, VID_UNDEFINED};
    Edge unweighted_max_edge = Edge{VID_UNDEFINED, VID_UNDEFINED};
    _v_range = VertexRange{1, 0};
    if (!edges.empty()) {
      auto minmax = find_min_max(execution::parallel{}, edges);
      const auto& min_edge = edges[minmax.first];
      const auto& max_edge = edges[minmax.second];
      unweighted_min_edge = Edge{min_edge.get_src(), min_edge.get_dst()};
      unweighted_max_edge = Edge{max_edge.get_src(), max_edge.get_dst()};
      _v_range = VertexRange{min_edge.get_src(), max_edge.get_src()};
    }

    _nb_local_vertices = (1ull + _v_range.second) - _v_range.first;
    REORDERING_BARRIER
    get_timer().stop("graph_internal_init_setup", round);
    REORDERING_BARRIER
    // SEQ_EX(ctx, PRINT_VAR(_v_range); PRINT_VAR(min_edge);
    // PRINT_VAR(max_edge););
    REORDERING_BARRIER
    get_timer().start("graph_internal_init_locators", round);
    REORDERING_BARRIER
    locator_ = LocatorWrapper(unweighted_min_edge, unweighted_max_edge, edges);
    REORDERING_BARRIER
    get_timer().stop("graph_internal_init_locators", round);
    REORDERING_BARRIER
    // if (ctx.rank() == 0)
    //   std::cout << locator_split << std::endl;
  }
  DistributedGraph(const DistributedGraph&) = delete;

  bool is_local(VId v) const {
    return (_v_range.first <= v) && (v <= _v_range.second);
  }

  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<!compactify_, VId> get_local_id(VId v) const {
    return v - _v_range.first;
  }

  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<compactify_, VId> get_local_id(VId v) const {
    auto it = compactification_data.global_id_to_local_id.find(v + 1);
    return (*it).second;
  }
  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<compactify_, VId>
  get_global_id(VId local_v) const {
    return compactification_data.local_id_to_global_id[local_v];
  }
  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<!compactify_, VId>
  get_global_id(VId local_v) const {
    return local_v + _v_range.first;
  }
  [[nodiscard]] VertexRange get_range() const { return _v_range; }
  [[nodiscard]] const container<EdgeType>& edges() const { return _edges; }
  [[nodiscard]] container<EdgeType>& edges() { return _edges; }
  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<!compactify_, std::size_t> local_n() const {
    return _nb_local_vertices;
  }
  template <bool compactify_ = compactify>
  [[nodiscard]] std::enable_if_t<compactify_, std::size_t> local_n() const {
    return compactification_data.num_compactified_vertices;
  }
  [[nodiscard]] const LocatorWrapper& locator() const { return locator_; }
  template <typename Comp> container<EdgeType> debug_get_edges(Comp&& comp) {
    auto tmp = edges();
    std::sort(tmp.begin(), tmp.end(), comp);
    return tmp;
  }

private:
  DistributedGraph_CompactificationData<compactify> compactification_data;
  container<EdgeType>& _edges;
  VertexRange _v_range;
  LocatorWrapper locator_;
  VId _nb_local_vertices;
  bool is_sparse_locator_used;
};
} // namespace hybridMST
