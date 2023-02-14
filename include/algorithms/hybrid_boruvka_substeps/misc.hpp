#pragma once

#include <execution>
#include <unordered_set>

#include "../../../external/graphs/external/KaDiS/include/AmsSort/Timer/Tracker.hpp"
#include "AmsSort/AmsSort.hpp"
#include "RQuick/RQuick.hpp"

//#include "algorithms/gbbs_reimplementation.hpp"
#include "algorithms/distributed_partitioning.hpp"
//#include "algorithms/local_contraction_local_edge_removal.hpp"
#include "datastructures/distributed_array.hpp"
#include "datastructures/distributed_graph.hpp"
#include "datastructures/distributed_parent_array.hpp"
#include "datastructures/growt.hpp"
//#include "datastructures/lookup_map.hpp"
#include "definitions.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mpi/scan.hpp"
#include "util/allocators.hpp"
#include "util/macros.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

namespace hybridMST {

template <typename Graph>
inline bool stop_boruvka(const Graph& graph, const std::size_t m_initial,
                         const std::size_t n_initial) {
  mpi::MPIContext ctx;
  const auto min_num_edges = mpi::allreduce_min(graph.edges().size());
  const auto max_num_edges = mpi::allreduce_max(graph.edges().size());
  const auto sum_num_edges = mpi::allreduce_sum(graph.edges().size());
  const auto min_num_vertices = mpi::allreduce_min(graph.local_n());
  const auto max_num_vertices = mpi::allreduce_max(graph.local_n());
  const auto sum_num_vertices = mpi::allreduce_sum(graph.local_n());
  const std::size_t vertex_stop_threshold = std::max(35'000, 2 * ctx.size());
  const bool stop_boruvka_ = sum_num_vertices < vertex_stop_threshold;
  if (ctx.rank() == 0) {
    std::cout << "Stop-Boruvka: " << std::boolalpha << stop_boruvka_
              << " min_num_vertices: " << min_num_vertices
              << " max_num_vertices: " << max_num_vertices
              << " sum_num_vertices: " << sum_num_vertices
              << " min_num_edges: " << min_num_edges
              << " max_num_edges: " << max_num_edges
              << " sum_num_edges: " << sum_num_edges << std::endl;
  }
  return stop_boruvka_;
}

struct RedistributeViaPartitioning {
  template <typename... Args>
  static void redistribute(std::vector<Args...>& edges) {
    using EdgeType = typename std::vector<Args...>::value_type;
    mpi::MPIContext ctx;
    auto& timer = get_timer();
    timer.start_phase("redistribution");
    const auto& count = timer.get_phase_add_count();
    timer.add_phase(
        "comm_volume", count, edges.size() * sizeof(EdgeType),
        {Timer::DatapointsOperation::MAX, Timer::DatapointsOperation::MAX_DIF});

    edges = partition(edges, SrcDstWeightOrder<EdgeType>{});
    timer.stop_phase();
  }
};

struct RedistributeViaRquickSorting {
  template <typename... Args>
  static void redistribute(std::vector<Args...>& edges) {
    using EdgeType = typename std::vector<Args...>::value_type;
    mpi::MPIContext ctx;
    mpi::TypeMapper<EdgeType> tm;
    // const int kway = 64;
    int tag = 100000;
    std::random_device rd;
    std::mt19937_64 gen(2 * ctx.rank() + 1);
    auto& timer = get_timer();
    timer.start_phase("redistribution");
    const auto& count = timer.get_phase_add_count();
    timer.add_phase(
        "sort_volume", count, edges.size() * sizeof(EdgeType),
        {Timer::DatapointsOperation::MAX, Timer::DatapointsOperation::MAX_DIF});

    const std::size_t num_bytes_sent = edges.size() * sizeof(EdgeType);
    timer.increment_phase_add_count();
    RQuick::sort(tm.get_mpi_datatype(), edges, tag, gen, ctx.communicator());
    const std::size_t num_bytes_recv = edges.size() * sizeof(EdgeType);
    get_communication_tracker().add_volume(num_bytes_sent, num_bytes_recv);
    timer.stop_phase();
  }
};

struct RedistributeViaAmsSorting {
  template <typename... Args>
  static void redistribute(std::vector<Args...>& edges) {
    using EdgeType = typename std::vector<Args...>::value_type;
    mpi::MPIContext ctx;
    mpi::TypeMapper<EdgeType> tm;
    // const int kway = 64;
    int tag = 100000;
    std::random_device rd;
    std::mt19937_64 gen(ctx.rank());
    auto& timer = get_timer();
    timer.start_phase("redistribution");
    const auto& count = timer.get_phase_add_count();
    Ams::DetailedTracker tracker;
    timer.add_phase(
        "sort_volume", count, edges.size() * sizeof(EdgeType),
        {Timer::DatapointsOperation::MAX, Timer::DatapointsOperation::MAX_DIF});
    timer.increment_phase_add_count();
    const std::size_t num_bytes_sent = edges.size() * sizeof(EdgeType);
    Ams::sortTrackerLevel(tm.get_mpi_datatype(), edges, 2, gen, tracker,
                          ctx.communicator(), SrcDstWeightOrder<EdgeType>{});
    const std::size_t num_bytes_recv = edges.size() * sizeof(EdgeType);
    get_communication_tracker().add_volume(num_bytes_sent, num_bytes_recv);
    timer.stop_phase();
    tracker.max(ctx.communicator());
  }
};

struct RedistributeViaSelectedSorter {
  template <typename... Args>
  static void redistribute(std::vector<Args...>& edges) {
    using EdgeType = typename std::vector<Args...>::value_type;
    mpi::MPIContext ctx;
    mpi::TypeMapper<EdgeType> tm;
    const std::size_t global_num_edges = mpi::allreduce_sum(edges.size(), ctx);
    const std::size_t avg_edges_per_pe = global_num_edges / ctx.size();
    const std::size_t threshold_for_rquick = 5'12;
    if (avg_edges_per_pe <= threshold_for_rquick) {
      std::vector<EdgeType> edges_(edges.size());
      std::copy(edges.begin(), edges.end(), edges_.begin());
      RedistributeViaRquickSorting::redistribute(edges_);
      edges.resize(edges_.size());
      std::copy(edges_.begin(), edges_.end(), edges.begin());
      return;
    }
    auto& timer = get_timer();
    timer.start_phase("redistribution");
    SrcDstWeightOrder<EdgeType> comp;
    twolevel_partition(edges, SrcDstWeightOrder<EdgeType>{});
    timer.stop_phase();
  }
};

struct GetMstEdge {
  static auto compute_edge_id_offsets(std::size_t num_local_ref_edges) {
    auto edge_id_offsets = mpi::allgather(num_local_ref_edges);
    std::exclusive_scan(edge_id_offsets.begin(), edge_id_offsets.end(),
                        edge_id_offsets.begin(), 0ull);
    return edge_id_offsets;
  }
  static PEID get_pe(const GlobalEdgeId& id, const auto& edge_offsets) {
    auto it = std::upper_bound(edge_offsets.begin(), edge_offsets.end(), id);
    --it;
    return std::distance(edge_offsets.begin(), it);
  }
  template <typename Cont1, typename Cont2>
  static std::vector<WEdge> execute(const Cont1& ref_edges,
                                    const Cont2& mst_edges_global_ids) {

    get_timer().start("send_mst_edges_back_communicate", 0);
    const mpi::MPIContext ctx;
    const auto edge_id_offsets = compute_edge_id_offsets(ref_edges.size());
    auto filter = False_Predicate{};
    auto transformer = [](const GlobalEdgeId& id, const std::size_t&) {
      return id;
    };
    auto dst_computer = [&](const GlobalEdgeId& id, const std::size_t&) {
      return get_pe(id, edge_id_offsets);
    };
    auto global_ids = mpi::alltoall_combined(mst_edges_global_ids, filter,
                                             transformer, dst_computer);

    get_timer().stop("send_mst_edges_back_communicate", 0);
    get_timer().start("send_mst_edges_back_lookup", 0);

    WEdgeList mst_edges(global_ids.buffer.size());

    auto task = [&](uint task_id) {
      const std::size_t m = global_ids.buffer.size();
      const std::size_t t = ctx.threads_per_mpi_process();
      const std::size_t begin = m / t * task_id;
      const std::size_t end = m / t * (task_id + 1) + m % t;
      for (std::size_t i = begin; i < end; ++i) {
        const auto& global_id = global_ids.buffer[i];
        const auto local_id = global_id - edge_id_offsets[ctx.rank()];
        const auto& edge = ref_edges[local_id];
        WEdge out_edge{edge.get_src(), edge.get_dst(),edge.get_weight()};
        mst_edges[i] = out_edge;
      }
    };

    // for (const auto& global_id : global_ids.buffer) {
    //   const auto local_id = global_id - edge_id_offsets[ctx.rank()];
    //   mst_edges.push_back(ref_edges[local_id]);
    // }
#pragma omp parallel for
    for (std::size_t i = 0; i < ctx.threads_per_mpi_process(); ++i) {
      task(i);
    }
    get_timer().stop("send_mst_edges_back_lookup", 0);

    return mst_edges;
  }
};

} // namespace hybridMST
