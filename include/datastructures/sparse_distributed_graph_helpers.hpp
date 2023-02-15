#pragma once

#include "mpi/allgather.hpp"
#include "mpi/alltoall_combined.hpp"
#include "mpi/twolevel_columnmajor_communicator.hpp"
#include "util/utils.hpp"

namespace hybridMST {
namespace sparse_graph {
struct EdgeInterval {
  Edge min_edge_;
  Edge max_edge_;
  EdgeInterval()
      : min_edge_{VID_UNDEFINED, VID_UNDEFINED}, max_edge_{VID_UNDEFINED,
                                                           VID_UNDEFINED} {}
  EdgeInterval(Edge min_edge, Edge max_edge)
      : min_edge_(min_edge), max_edge_(max_edge) {}
  bool is_empty() const { return !is_defined(min_edge_.get_src()); }
  friend std::ostream& operator<<(std::ostream& out,
                                  const EdgeInterval& interval) {
    return out << "[" << interval.min_edge_ << " - " << interval.max_edge_
               << "]";
  }
};

struct EdgeIntervalPE {
  EdgeIntervalPE() = default;
  EdgeIntervalPE(EdgeInterval interval_, PEID pe_)
      : interval{interval_}, pe{pe_} {}
  EdgeInterval interval;
  PEID pe;
  friend std::ostream& operator<<(std::ostream& out,
                                  const EdgeIntervalPE& interval_pe) {
    return out << interval_pe.interval << " on pe: " << interval_pe.pe;
  }
};

struct EdgeIntervalColumn {
  EdgeIntervalColumn() = default;
  EdgeIntervalColumn(EdgeInterval interval_, PEID column_)
      : interval{interval_}, column{column_} {}
  EdgeInterval interval;
  PEID column;
  friend std::ostream& operator<<(std::ostream& out,
                                  const EdgeIntervalColumn& interval_column) {
    return out << interval_column.interval
               << " on column: " << interval_column.column;
  }
};

struct Reply {
  Reply() = default;
  Reply(Edge requested_edge_, PEID pe_, unsigned char is_split_)
      : requested_edge{requested_edge_}, pe{pe_}, is_split{is_split_} {}
  Edge requested_edge;
  PEID pe;
  unsigned char is_split;
};
struct EdgeHasher {
  std::size_t operator()(const Edge& edge) const {
    const static auto hasher = std::hash<VId>{};
    return hasher(edge.get_src()) ^ (hasher(edge.get_dst()) << 1);
  }
};

class VertexLocator {
private:
  auto get_it_to_min_pe_entry(const Edge& edge) const {
    assert(get_min_column_or_sentinel(edge) ==
           twolevel_communicators_.get()
               .get_row_ctx()
               .rank()); // edge is requested on correct column
    const auto it = std::lower_bound(
        non_empty_edge_intervals_.begin(), non_empty_edge_intervals_.end(),
        edge, [](const EdgeIntervalPE& interval_pe, const Edge& comp_arg_edge) {
          return interval_pe.interval.max_edge_ < comp_arg_edge;
        });
    return it;
  }

public:
  struct PESplitStatus {
    PEID pe;
    bool is_split;
    friend std::ostream& operator<<(std::ostream& out,
                                    const PESplitStatus& elem) {
      return out << elem.pe << " is split: " << elem.is_split;
    }
    bool operator!=(const PESplitStatus& other) const {
      return std::tie(pe, is_split) != std::tie(other.pe, other.is_split);
    }
  };

  VertexLocator()
      : rank{mpi::MPIContext{}.rank()},
        twolevel_communicators_{mpi::get_twolevel_columnmajor_communicators()} {
  }
  template <typename Edges>
  VertexLocator(Edge min_edge, Edge max_edge, const Edges& edges)
      : VertexLocator{} {
    init(min_edge, max_edge, edges);
  }
  /// First PE local [min_edge, max_edge] intervals are computed.
  /// These are then exchanged columnwise.
  template <typename Edges>
  void init(Edge min_edge, Edge max_edge, const Edges& edges) {
    min_edge_ = min_edge;
    max_edge_ = max_edge;

    const auto& column_ctx = twolevel_communicators_.get().get_col_ctx();
    const auto& row_ctx = twolevel_communicators_.get().get_row_ctx();
    EdgeInterval local_interval{min_edge, max_edge};
    // column-wise exchange of [min_edge, max_edge]
    initial_edge_intervals_ = mpi::allgather(local_interval, column_ctx);
    non_empty_edge_intervals_.reserve(column_ctx.size());
    for (int i = 0;
         static_cast<std::size_t>(i) < initial_edge_intervals_.size(); ++i) {
      const auto& interval = initial_edge_intervals_[i];
      if (!interval.is_empty())
        non_empty_edge_intervals_.emplace_back(
            interval,
            twolevel_communicators_.get().min_org_rank_of_column() + i);
    }
    // edge intervals is already sorted by construction as the world
    // communicator is split in column-major order
    const static Edge sentinel_edge{VID_UNDEFINED, VID_UNDEFINED};
    const Edge min_edge_on_column =
        non_empty_edge_intervals_.empty()
            ? sentinel_edge
            : non_empty_edge_intervals_.front().interval.min_edge_;
    const Edge max_edge_on_column =
        non_empty_edge_intervals_.empty()
            ? sentinel_edge
            : non_empty_edge_intervals_.back().interval.max_edge_;
    const EdgeInterval interval_on_column{min_edge_on_column,
                                          max_edge_on_column};
    const auto column_edge_intervals = mpi::row_wise_allgatherv_on_column_data(
        interval_on_column, twolevel_communicators_.get());

    non_empty_column_edge_intervals_.reserve(row_ctx.size());
    for (int i = 0; static_cast<std::size_t>(i) < column_edge_intervals.size();
         ++i) {
      const auto& interval = column_edge_intervals[i];
      if (!interval.is_empty()) {
        non_empty_column_edge_intervals_.emplace_back(interval, i);
        if (i == row_ctx.rank()) {
          idx_own_entry_column_edge_interval_ =
              non_empty_column_edge_intervals_.size() - 1;
        }
      }
    }
    request_ghost_locations(edges);
  }
  [[nodiscard]] bool is_home_of_v_min() const {
    if (!is_defined(min_edge_.get_src())) {
      return true;
    }
    const auto it = get_it_to_min_pe_entry(min_edge_);
    const auto pe_idx = std::distance(non_empty_edge_intervals_.begin(), it);
    return (it->pe == rank) &&
           !has_predecessor_pe_same_vertex(min_edge_, pe_idx);
  }
  [[nodiscard]] bool is_home_of_v_max() const {
    if (!is_defined(max_edge_.get_src())) {
      return true;
    }
    const auto it = get_it_to_min_pe_entry(max_edge_);
    const auto pe_idx = std::distance(non_empty_edge_intervals_.begin(), it);
    return (it->pe == rank) && !has_successor_pe_same_vertex(max_edge_, pe_idx);
  }
  [[nodiscard]] bool is_v_min_split() const {
    if (!is_defined(min_edge_.get_src())) {
      return false;
    }
    const auto it = get_it_to_min_pe_entry(min_edge_);
    const auto pe_idx = std::distance(non_empty_edge_intervals_.begin(), it);
    const bool is_split = has_predecessor_pe_same_vertex(min_edge_, pe_idx) ||
                          has_successor_pe_same_vertex(min_edge_, pe_idx);
    return is_split;
  }
  [[nodiscard]] bool is_v_max_split() const {
    if (!is_defined(max_edge_.get_src())) {
      return false;
    }
    const auto it = get_it_to_min_pe_entry(max_edge_);
    const auto pe_idx = std::distance(non_empty_edge_intervals_.begin(), it);
    const bool is_split = has_predecessor_pe_same_vertex(max_edge_, pe_idx) ||
                          has_successor_pe_same_vertex(max_edge_, pe_idx);
    return is_split;
  }
  [[nodiscard]] bool is_local(VId v) const {
    return (min_edge_.get_src() <= v) && (v <= max_edge_.get_src());
  }

  [[nodiscard]] VId v_min() const { return min_edge_.get_src(); }

  [[nodiscard]] VId v_max() const { return max_edge_.get_src(); }

  [[nodiscard]] PESplitStatus
  get_min_pe_and_split_info(const Edge& edge) const {
    const auto it = edge_pe.find(edge);
    return it->second;
  }
  [[nodiscard]] PESplitStatus
  get_min_pe_and_split_info_or_sentinel(const Edge& edge,
                                        PEID sentinel = -1) const {
    const auto it = edge_pe.find(edge);
    if (it == edge_pe.end()) {
      return PESplitStatus{sentinel, false};
    }
    return it->second;
  }
  [[nodiscard]] PEID get_min_pe(const Edge& edge) const {
    return get_min_pe_and_split_info(edge).pe;
  }
  [[nodiscard]] PEID get_min_pe_or_sentinel(const Edge& edge,
                                            PEID sentinel = -1) const {
    return get_min_pe_and_split_info_or_sentinel(edge, sentinel).pe;
  }

  [[nodiscard]] PEID get_min_pe(const VId& v) const {
    return get_min_pe(sentinel_edge(v));
  }

  [[nodiscard]] PESplitStatus get_min_pe_and_split_info(const VId& v) {
    return get_min_pe_and_split_info(sentinel_edge(v));
  }
  std::string debug_print() const {
    mpi::MPIContext ctx;
    std::stringstream ss;
    ss << "rank: " << ctx.rank() << " min_edge: " << min_edge_
       << " max_edge: " << max_edge_ << "\n";
    ss << "non_empty_edge_intervals: ";
    for (const auto& elem : non_empty_edge_intervals_)
      ss << "\n\t" << elem;
    ss << "\nnon_empty_column_intervals: ";
    for (const auto& elem : non_empty_column_edge_intervals_)
      ss << "\n\t" << elem << " ";
    return ss.str();
  }

private:
  PEID get_min_column_or_sentinel(const Edge& edge, PEID sentinel = -1) const {
    const auto it = std::lower_bound(
        non_empty_column_edge_intervals_.begin(),
        non_empty_column_edge_intervals_.end(), edge,
        [](const EdgeIntervalColumn& interval_column,
           const Edge& comp_arg_edge) {
          return interval_column.interval.max_edge_ < comp_arg_edge;
        });
    const bool points_to_end = it == non_empty_column_edge_intervals_.end();
    const PEID column = !points_to_end ? it->column : sentinel;
    return column;
  }

  Edge sentinel_edge(VId src) const { return Edge{src, VID_UNDEFINED}; }

  template <typename Edges> void request_ghost_locations(Edges& edges) {
    mpi::MPIContext ctx;
    struct EdgeRequestingPE {
      Edge edge;
      PEID pe;
    };
    auto filter = [&](const Edge& edge, const std::size_t&) {
      return get_min_column_or_sentinel(edge, -1) == -1;
    };
    auto transformer = [&](const Edge& edge, const std::size_t&) {
      return EdgeRequestingPE{edge, ctx.rank()};
    };
    auto dst_calculator = [&](const Edge& edge, const std::size_t& i) {
      const auto& target_col_id = get_min_column_or_sentinel(edge);
      const auto& target_col_size =
          twolevel_communicators_.get().size_of_col_with(target_col_id);
      const auto& min_world_rank_target_column =
          twolevel_communicators_.get().min_org_rank_of_column(target_col_id);
      return min_world_rank_target_column +
             (i % target_col_size); // add true randomness if load balancing is
                                    // not sufficient
    };
    auto requests =
        mpi::alltoall_combined(edges, filter, transformer, dst_calculator);
    auto reply_filter = False_Predicate{};
    auto reply_transformer = [&](const EdgeRequestingPE& edge_requesting_pe, const std::size_t&) {
      return process_requested_edge(edge_requesting_pe.edge);
    };
    auto reply_dst_calculator = [&](const EdgeRequestingPE& edge_requesting_pe, const std::size_t&) {
      return edge_requesting_pe.pe;
    };
    // SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(requests.buffer););
    auto replies = mpi::alltoall_combined(
        requests.buffer, reply_filter, reply_transformer, reply_dst_calculator);
    for (const auto& reply : replies.buffer) {
      const auto pe_split_status =
          PESplitStatus{reply.pe, static_cast<bool>(reply.is_split)};
      edge_pe[reply.requested_edge] = pe_split_status;
      edge_pe[sentinel_edge(reply.requested_edge.get_src())] = pe_split_status;
    }
  }

  [[nodiscard]] bool has_predecessor_pe_same_vertex(
      const Edge& edge, const std::size_t pe_idx_in_edge_intervals) const {
    if (pe_idx_in_edge_intervals > 0) {
      const auto& predecessor_interval =
          non_empty_edge_intervals_[pe_idx_in_edge_intervals - 1];
      return predecessor_interval.interval.max_edge_.get_src() ==
             edge.get_src();
    }
    const auto& column_idx = get_own_entry_idx_in_column_edge_interval();
    if (!is_defined(column_idx)) {
      PRINT_WARNING_AND_ABORT("requested info from empty column");
    }
    if (column_idx > 0) {
      const auto& predecessor_column_interval =
          non_empty_column_edge_intervals_[column_idx - 1];
      return predecessor_column_interval.interval.max_edge_.get_src() ==
             edge.get_src();
    }
    return false;
  }

  [[nodiscard]] bool has_successor_pe_same_vertex(
      const Edge& edge, const std::size_t pe_idx_in_edge_intervals) const {
    if (pe_idx_in_edge_intervals + 1 < non_empty_edge_intervals_.size()) {
      const auto& successor_interval =
          non_empty_edge_intervals_[pe_idx_in_edge_intervals + 1];
      return successor_interval.interval.min_edge_.get_src() == edge.get_src();
    }
    const auto& column_idx = get_own_entry_idx_in_column_edge_interval();
    if (!is_defined(column_idx)) {
      PRINT_WARNING_AND_ABORT("requested info from empty column");
    }
    if (column_idx + 1 < non_empty_column_edge_intervals_.size()) {
      const auto& successor_column_interval =
          non_empty_column_edge_intervals_[column_idx + 1];
      return successor_column_interval.interval.min_edge_.get_src() ==
             edge.get_src();
    }
    return false;
  }

  Reply process_requested_edge(const Edge& edge) const {
    // mpi::MPIContext ctx;
    const auto it = get_it_to_min_pe_entry(edge);
    const auto pe_idx = std::distance(non_empty_edge_intervals_.begin(), it);

    // std::stringstream ss;
    // ss << ctx.rank() << " " << edge << " " << pe_idx << std::endl;
    // std::cout << ss.str() << std::endl;
    const bool is_split = has_predecessor_pe_same_vertex(edge, pe_idx) ||
                          has_successor_pe_same_vertex(edge, pe_idx);
    Reply reply;
    reply.requested_edge = edge;
    reply.pe = it->pe;
    reply.is_split = is_split;
    return reply;
  }

  std::size_t get_own_entry_idx_in_column_edge_interval() const {
    return idx_own_entry_column_edge_interval_;
  }

  int rank;
  std::reference_wrapper<mpi::TwoLevelColumnMajorCommunicator>
      twolevel_communicators_;
  std::vector<EdgeInterval> initial_edge_intervals_;
  std::vector<EdgeIntervalPE> non_empty_edge_intervals_;
  std::vector<EdgeIntervalColumn> non_empty_column_edge_intervals_;
  VId idx_own_entry_column_edge_interval_ = VID_UNDEFINED;
  std::unordered_map<Edge, PESplitStatus, EdgeHasher> edge_pe;
  Edge min_edge_;
  Edge max_edge_;
};
} // namespace sparse_graph
} // namespace hybridMST
