
#include "RQuick/RQuick.hpp"

#include "algorithms/hybrid_boruvka_re_computations.hpp"
#include "datastructures/distributed_graph.hpp"
#include "definitions.hpp"
#include "util/timer.hpp"

namespace hybridMST {
namespace external {

template <typename Graph>
inline bool stop_boruvka(const Graph& graph, const std::size_t m_initial,
                         const std::size_t n_initial) {
  mpi::MPIContext ctx;
  const auto min_num_edges = mpi::allreduce_min(graph.edges().size());
  const auto min_num_vertices = mpi::allreduce_min(graph.local_n());
  auto v_range = graph.get_range();
  const auto min_num_extended_vertices =
      mpi::allreduce_min(v_range.second - v_range.first);
  if (ctx.rank() == 0) {
    std::cout << "Stop-Overview: " << min_num_extended_vertices << " "
              << min_num_vertices << " " << min_num_edges << std::endl;
  }
  bool edge_min_reached = (min_num_edges * ctx.size()) < m_initial;
  bool vertex_min_reached = (min_num_vertices * ctx.size()) < n_initial;
  return edge_min_reached || vertex_min_reached;
}
struct RedistributeViaSorting {
  template <typename... Args>
  static void redistribute(std::vector<Args...>& edges) {
    using EdgeType = typename std::vector<Args...>::value_type;
    mpi::MPIContext ctx;
    mpi::TypeMapper<EdgeType> tm;
    // const int kway = 64;
    int tag = 100000;
    std::random_device rd;
    std::mt19937_64 gen(rd());
    auto& timer = get_timer();
    timer.start_phase("redistribution");
    const auto& count = timer.get_phase_add_count();
    timer.add_phase(
        "sort_volume", count, edges.size() * sizeof(EdgeType),
        {Timer::DatapointsOperation::MAX, Timer::DatapointsOperation::MAX_DIF});
    timer.increment_phase_add_count();
    RQuick::sort(tm.get_mpi_datatype(), edges, tag, gen, ctx.communicator());
    timer.stop_phase();
  }
};

std::vector<WEdgeId> local_kernelization(WEdgeList& edges,
                                         const VertexRange& range,
                                         VId initial_n, VId initial_m,
                                         const mpi::MPIContext ctx) {
  get_timer().start("barrier0", 0);
  ctx.barrier();
  get_timer().stop("barrier0", 0);

  get_timer().start("local_kernelization");
  get_timer().add("graph_edges_initial", 0, initial_m,
                  Timer::DatapointsOperation::ID);
  get_timer().add("graph_vertices_initial", 0, initial_n,
                  Timer::DatapointsOperation::ID);
  auto augmented_edges =
      LocalKernelization_LocalEdgeRemoval::execute(range, edges);
  get_timer().add("graph_edges_kernel", 0, augmented_edges.size(),
                  Timer::DatapointsOperation::ID);
  get_timer().add("graph_vertices_kernel", 0, initial_n,
                  Timer::DatapointsOperation::ID);
  get_timer().stop("local_kernelization");
  return augmented_edges;
}

non_init_vector<LocalEdgeId> compute_min_edges(std::size_t i,
                                               DistributedGraph<false>& graph,
                                               const mpi::MPIContext& ctx) {
  get_timer().start("barrier2", i);
  ctx.barrier();
  get_timer().stop("barrier2", i);
  get_timer().start("min_edges", i);
  auto min_edge_ids_par = MinimumEdgeTBB::execute(graph);
  get_timer().stop("min_edges", i);
  return min_edge_ids_par;
}

std::vector<uint8_t>
is_representative(std::size_t i, non_init_vector<LocalEdgeId>& min_edge_ids_par,
                  DistributedGraph<false>& graph, const mpi::MPIContext& ctx) {
  get_timer().start("barrier3", i);
  ctx.barrier();
  get_timer().stop("barrier3", i);
  get_timer().start("is_representative", i);
  auto is_rep =
      IsRepresentative_Push::compute_representatives(min_edge_ids_par, graph);
  get_timer().stop("is_representative", i);
  return is_rep;
}

void add_mst_edges(std::size_t i, DistributedGraph<false>& graph,
                   const std::vector<uint8_t>& is_rep,
                   const non_init_vector<LocalEdgeId>& min_edge_ids_par,
                   non_init_vector<GlobalEdgeId>& mst_edge_ids,
                   const mpi::MPIContext& ctx) {
  get_timer().start("barrier4", i);
  ctx.barrier();
  get_timer().stop("barrier4", i);
  get_timer().start("add_mst_edges", i);
  AddMstEdgesSeq::execute(graph, is_rep, min_edge_ids_par, mst_edge_ids);
  get_timer().stop("add_mst_edges", i);
}
non_init_vector<VId>
compute_representatives(std::size_t i, DistributedGraph<false>& graph,
                        const std::vector<uint8_t>& is_rep,
                        const non_init_vector<LocalEdgeId>& min_edge_ids_par,
                        const mpi::MPIContext& ctx) {
  get_timer().start("barrier5", i);
  ctx.barrier();
  get_timer().stop("barrier5", i);
  get_timer().start("compute_representatives", i);
  auto rep_local_vertices = ComputeRepresentative::compute_representatives(
      graph, is_rep, min_edge_ids_par);
  get_timer().stop("compute_representatives", i);
  return rep_local_vertices;
}

void get_new_names(std::size_t i, 
    DistributedGraph<false>& graph,
tbb::concurrent_hash_map<VId, VId>& map,
              const non_init_vector<VId>& rep_local_vertices,
              const mpi::MPIContext& ctx) {
  get_timer().start("barrier6", i);
  ctx.barrier();
  get_timer().stop("barrier6", i);
  get_timer().start("get_ghost_representatives", i);
  map = ExchangeRepresentativesRequest::exchange_reps_with_seq_filter(
          graph, rep_local_vertices);
  get_timer().stop("get_ghost_representatives", i);
}

void rename_edges(
    std::size_t i, DistributedGraph<false>& graph,
    const non_init_vector<VId>& rep_local_vertices,
    tbb::concurrent_hash_map<VId, VId>& name_newName_ghost_vertices,
    const mpi::MPIContext& ctx) {
  get_timer().start("barrier7", i);
  ctx.barrier();
  get_timer().stop("barrier7", i);
  get_timer().start("rename_edges", i);
  EdgeRenamer::rename_edges(graph, rep_local_vertices,
                            name_newName_ghost_vertices);
  get_timer().stop("rename_edges", i);
}

void redistribute(std::size_t i, DistributedGraph<false>& graph,
                  const mpi::MPIContext& ctx) {
  get_timer().start("barrier8", i);
  ctx.barrier();
  get_timer().stop("barrier8", i);
  get_timer().start("redistribute_edges", i);
  RedistributeViaSorting::redistribute(graph.edges());
  get_timer().stop("redistribute_edges", i);
}

void remove_duplicate(std::size_t i, DistributedGraph<false>& graph,
                      const mpi::MPIContext& ctx) {
  get_timer().start("barrier9", i);
  ctx.barrier();
  get_timer().stop("barrier9", i);
  get_timer().start("remove_duplicate_edges", i);
  EdgeProcessor::remove_duplicates(graph, i + 10);
  get_timer().stop("remove_duplicate_edges", i);
}

bool stop_criterion_call(std::size_t i, const DistributedGraph<false>& graph,
                         VId initial_m, VId initial_n,
                         const mpi::MPIContext& ctx) {
  get_timer().start("barrier10", i);
  ctx.barrier();
  get_timer().stop("barrier10", i);
  get_timer().start("stop_criterion", i);
  bool stop_boruvka_ = stop_boruvka(graph, initial_m, initial_n);
  get_timer().stop("stop_criterion", i);
  return stop_boruvka_;
}

WEdgeList after_loop(std::vector<WEdgeId>& augmented_edges,
                     non_init_vector<GlobalEdgeId>& mst_edge_ids,
                     const std::vector<WEdge>& edges,
                     const mpi::MPIContext& ctx) {
  get_timer().start("base_case");
  get_timer().start("gather_on_root");
  auto recv_edges = hybridMST::mpi::gatherv(augmented_edges.data(),
                                            augmented_edges.size(), 0, ctx);

  get_timer().add("graph_edges_base_case", 0, recv_edges.size(),
                  Timer::DatapointsOperation::MAX);
  augmented_edges.clear();
  augmented_edges.shrink_to_fit();

  get_timer().stop("gather_on_root");
  get_timer().start("gbbs");
  std::vector<GlobalEdgeId> ids_gbbs;
  const std::size_t n = recv_edges.empty() ? 0ull : src(recv_edges.back()) + 1;
  // local_kruskal(n, recv_edges, ids_gbbs, hybridMST::execution::parallel{});
  gbbs_reimplementation(n, recv_edges, ids_gbbs);
  get_timer().stop("gbbs");

  auto mst_edge_ids_combined = combine(mst_edge_ids, ids_gbbs);
  get_timer().stop("base_case");
  get_timer().start("send_mst_edges_back");
  const auto mst_edges = GetMstEdge::execute(edges, mst_edge_ids_combined);
  get_timer().stop("send_mst_edges_back");
  return mst_edges;
}
} // namespace external
} // namespace hybridMST
