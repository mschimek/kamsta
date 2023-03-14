#pragma once

#include "datastructures/growt.hpp"
#include "definitions.hpp"
#include "is_representative_computation.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "mpi/twolevel_alltoall.hpp"

#include "datastructures/distributed_array.hpp"
#include "datastructures/distributed_parent_array.hpp"

namespace hybridMST {
struct ComputeRepresentative_Base {
  static VId set_msb(VId v) {
    constexpr VId zero = 0u;
    constexpr VId ones = ~zero;
    constexpr VId all_ones_but_msb = ones >> 1;
    constexpr VId only_msb_set = ~all_ones_but_msb;
    return v | only_msb_set;
  }

  static VId reset_msb(VId v) {
    constexpr VId zero = 0u;
    constexpr VId ones = ~zero;
    constexpr VId all_ones_but_msb = ones >> 1;
    return v & all_ones_but_msb;
  }

  static bool is_msb_set(VId v) {
    constexpr VId zero = 0u;
    constexpr VId ones = ~zero;
    constexpr VId all_ones_but_msb = ones >> 1;
    constexpr VId only_msb_set = ~all_ones_but_msb;
    return v & only_msb_set;
  }

  template <typename Alloc1, typename Alloc2, typename Graph>
  static inline VId
  consecutive_vertex_ids(const Graph& graph,
                         const std::vector<std::uint8_t, Alloc1>& is_rep,
                         std::vector<VId, Alloc2>& predecessors) {
    mpi::MPIContext ctx;
    auto& locator = graph.split_locator();
    std::size_t num_reps = parlay::count_if(
        is_rep, [](const std::uint8_t& i) { return (i == 1); });
    if (is_rep.size() == 0) {
      mpi::exscan_sum(num_reps, ctx);
      return mpi::allreduce_sum(num_reps);
    }
    if (!locator.is_home_of_v_min)
      --num_reps;
    std::size_t global_prefix = mpi::exscan_sum(num_reps, ctx);
    const VId new_num_vertices = mpi::allreduce_sum(num_reps);
    if (ctx.rank() == 0)
      global_prefix = 0;
    // treat first vertex specially
    predecessors[0] = global_prefix;
    if (const VId local_id_v_min = graph.get_local_id(locator.v_min);
        !locator.is_home_of_v_min) {

      predecessors[local_id_v_min] = global_prefix - 1;
    } else if (is_rep[local_id_v_min] == 1) {
      predecessors[local_id_v_min] = global_prefix++;
    }

    // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(predecessors););
    assign_initialize(predecessors.begin() + 1, predecessors.end(),
                      [&](const std::size_t& i) {
                        return (is_rep[i + 1] == 1) ? 1ull : 0ull;
                      });
    // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(predecessors););
    // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(is_rep););
    auto s = parlay::slice(predecessors.begin() + 1, predecessors.end());
    parlay::scan_inplace(s);
    // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(predecessors););
    map(predecessors.begin() + 1, predecessors.end(),
        [&](VId& v, const std::size_t&) { v += global_prefix; });
    // SEQ_EX(ctx, PRINT_VECTOR_WITH_INDEX(predecessors););
    //  predecessors = predecessors2;

    return new_num_vertices;
  }

  template <typename Predecessors, typename Graph>
  static inline void
  initialize_predecessors(const Graph& graph,
                          const non_init_vector<LocalEdgeId>& min_edge_idxs,
                          Predecessors& predecessors) {
    parallel_for(0, predecessors.size(), [&](std::size_t i) {
      const LocalEdgeId idx = min_edge_idxs[i];
      if (is_defined(idx)) {
        const VId& dst_id = graph.edges()[idx].get_src(); //@TODO is this right?
        predecessors[i] = dst_id;
      }
    });
  }
};

struct ComputeRepresentative : public ComputeRepresentative_Base {

  static bool is_jumping_finished(std::size_t nb_vertices_to_get_rooted,
                                  const hybridMST::mpi::MPIContext& ctx) {
    return allreduce_max(nb_vertices_to_get_rooted, ctx) == 0;
  }
  struct VertexPredecessorPePred {
    VId requested_vertex;
    VId predecessor;
    PEID pe_of_predecessor;
  };
  template <typename Graph>
  static bool pointer_jumping_round_two_level(
      const Graph& graph, const Span<VId>& vertices_to_query_local_id,
      non_init_vector<VId>& predecessors,
      non_init_vector<PEID>& home_pe_predecessors, unsigned int round) {
    mpi::MPIContext ctx;
    get_timer().start("reps_jumping_jumping_inner", round);
    {
      asm volatile("" ::: "memory");
      get_timer().start("reps_jumping_jumping_inner_inner", round);
      if (is_jumping_finished(vertices_to_query_local_id.size(), ctx)) {
        get_timer().start("reps_jumping_jumping_inner_inner", round);
        get_timer().stop("reps_jumping_jumping_inner", round);
        return false;
      }
      auto filter = [&](const VId& idx, const std::size_t) {
        return ComputeRepresentative::is_msb_set(predecessors[idx]);
      };
      auto transformer = [&](const VId& idx, const std::size_t) {
        return Edge{idx, predecessors[idx]};
      };
      auto dst_computer = [&](const VId& idx, const std::size_t) {
        return home_pe_predecessors[idx];
      };
      auto request = mpi::two_level_alltoall(vertices_to_query_local_id, filter,
                                             transformer, dst_computer);
      auto filter_reply = False_Predicate{};
      auto transformer_reply = [&](const auto& elem, const std::size_t) {
        const Edge& edge = elem.payload();
        const auto local_id = graph.get_local_id(edge.get_dst());
        return VertexPredecessorPePred{edge.get_src(), predecessors[local_id],
                                       home_pe_predecessors[local_id]};
      };
      auto return_sender = [](const auto& elem, std::size_t) {
        return elem.get_sender();
      };
      auto reply = mpi::two_level_alltoall_extract(
          request.buffer, filter_reply, transformer_reply, return_sender);

      get_timer().start("reps_jumping_jumping_inner_write", round);
      if (ctx.rank() == 0) {
        std::cout << " received reply: " << round << std::endl;
      }
      parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
        const auto& [src, predecessor, pe_of_predecessor] = reply.buffer[i];
        predecessors[src] = predecessor;
        home_pe_predecessors[src] = pe_of_predecessor;
      });
      get_timer().stop("reps_jumping_jumping_inner_write", round);
      get_timer().stop("reps_jumping_jumping_inner_inner", round);
      asm volatile("" ::: "memory");
    }
    get_timer().stop("reps_jumping_jumping_inner", round);
    return true; // another round is needed
  }
  template <typename Graph>
  static bool pointer_jumping_round(const Graph& graph,
                                    const Span<VId>& vertices_to_query_local_id,
                                    non_init_vector<VId>& predecessors,
                                    unsigned int round) {
    mpi::MPIContext ctx;
    get_timer().start("reps_jumping_jumping_inner", round);
    {
      asm volatile("" ::: "memory");
      get_timer().start("reps_jumping_jumping_inner_inner", round);
      if (is_jumping_finished(vertices_to_query_local_id.size(), ctx)) {
        get_timer().start("reps_jumping_jumping_inner_inner", round);
        get_timer().stop("reps_jumping_jumping_inner", round);
        return false;
      }
      auto filter = [&](const VId& idx, const std::size_t) {
        return ComputeRepresentative::is_msb_set(predecessors[idx]);
      };
      auto transformer = [&](const VId& idx, const std::size_t) {
        return Edge{idx, predecessors[idx]};
      };
      auto dst_computer = [&](const VId& idx, const std::size_t) {
        return graph.split_locator().get_min_pe(predecessors[idx]);
      };

      const bool use_dense = true;
      int tag = round;

      auto request = mpi::twopass_alltoallv_openmp_special(
          vertices_to_query_local_id, filter, transformer, dst_computer,
          ctx.size(), ctx.threads_per_mpi_process(), use_dense, tag);

      auto filter_reply = False_Predicate{};
      auto transformer_reply = [&](const Edge& edge, const std::size_t) {
        const auto local_id = graph.get_local_id(edge.get_dst());
        return Edge{edge.get_src(), predecessors[local_id]};
      };
      auto dst_computer_reply = [&](const Edge&, const std::size_t i) {
        return request.get_pe(i);
      };
      ctx.barrier();
      if (ctx.rank() == 0)
        std::cout << " sent request: " << round << std::endl;

      auto reply = mpi::twopass_alltoallv_openmp_special(
          request.buffer, filter_reply, transformer_reply, dst_computer_reply,
          ctx.size(), ctx.threads_per_mpi_process(), use_dense, ++tag);
      ctx.barrier();
      get_timer().start("reps_jumping_jumping_inner_write", round);
      if (ctx.rank() == 0)
        std::cout << " received reply: " << round << std::endl;
      parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
        const auto& [src, dst] = reply.buffer[i];
        predecessors[src] = dst;
      });
      get_timer().stop("reps_jumping_jumping_inner_write", round);
      get_timer().stop("reps_jumping_jumping_inner_inner", round);
      asm volatile("" ::: "memory");
    }
    get_timer().stop("reps_jumping_jumping_inner", round);
    return true; // another round is needed
  }

  template <typename Graph>
  static bool pointer_jumping_round_filter(
      const Graph& graph,
      const non_init_vector<VId>& vertices_to_query_local_id,
      non_init_vector<VId>& predecessors, unsigned int round) {
    mpi::MPIContext ctx;
    get_timer().start("reps_jumping_jumping_inner", round);
    {
      if (is_jumping_finished(vertices_to_query_local_id.size(), ctx)) {
        get_timer().stop("reps_jumping_jumping_inner", round);
        return false;
      }

      const std::size_t num_queries = vertices_to_query_local_id.size();

      parlay::hashtable<parlay::hash_numeric<VId>> table(
          num_queries, parlay::hash_numeric<VId>{});

      parallel_for(0, num_queries, [&](std::size_t i) {
        const VId local_id_to_query = vertices_to_query_local_id[i];
        const VId predecessor = predecessors[local_id_to_query];
        if (!ComputeRepresentative::is_msb_set(predecessor)) {
          table.insert(predecessor);
        }
      });
      auto unique_predecessors = table.entries();
      SEQ_EX(ctx, PRINT_VECTOR(unique_predecessors.size());
             PRINT_VECTOR(vertices_to_query_local_id.size()););

      auto filter = False_Predicate{};
      auto transformer = [&](const VId& predecessor, const std::size_t) {
        return predecessor;
      };
      auto dst_computer = [&](const VId& predecessor, const std::size_t) {
        return graph.split_locator().get_min_pe(predecessor);
      };

      auto request = mpi::twopass_alltoallv_openmp_special(
          unique_predecessors, filter, transformer, dst_computer, ctx.size(),
          ctx.threads_per_mpi_process());

      auto filter_reply = False_Predicate{};
      auto transformer_reply = [&](const VId& requested_v, const std::size_t) {
        const auto local_id = graph.get_local_id(requested_v);
        return Edge{requested_v, predecessors[local_id]};
      };
      auto dst_computer_reply = [&](const VId&, const std::size_t i) {
        return request.get_pe(i);
      };

      auto reply = mpi::twopass_alltoallv_openmp_special(
          request.buffer, filter_reply, transformer_reply, dst_computer_reply,
          ctx.size(), ctx.threads_per_mpi_process());

      growt::GlobalVIdMap<VId> grow_map{reply.buffer.size() * 1.2};
      parallel_for(0, reply.buffer.size(), [&](std::size_t i) {
        const auto [requested_predecessor, replied_predecessor] =
            reply.buffer[i];
        const auto [it, _] =
            grow_map.insert(requested_predecessor + 1, replied_predecessor);
        if (it == grow_map.end()) {
          std::cout << "growt wrong insert" << std::endl;
          std::abort();
        }
      });
      // SEQ_EX(ctx, PRINT_VECTOR(predecessors););
      parallel_for(0, vertices_to_query_local_id.size(), [&](std::size_t i) {
        const VId local_id_to_query = vertices_to_query_local_id[i];
        const VId& predecessor = predecessors[local_id_to_query];
        if (!ComputeRepresentative::is_msb_set(predecessor)) {
          auto it = grow_map.find(predecessor + 1);
          predecessors[local_id_to_query] = (*it).second;
        }
      });
      // SEQ_EX(ctx, PRINT_VECTOR(vertices_to_query_local_id);
      // PRINT_VECTOR(predecessors); PRINT_VECTOR(reply.buffer););
    }

    get_timer().stop("reps_jumping_jumping_inner", round);
    return true; // another round is needed
  }

  template <typename Graph>
  static non_init_vector<VId>
  compute_vertices_to_query(const Graph& graph,
                            non_init_vector<VId>& predecessors,
                            const non_init_vector<uint8_t>& is_representative,
                            const non_init_vector<LocalEdgeId>& min_edge_idxs) {
    mpi::MPIContext ctx;
    const int num_threads = ctx.threads_per_mpi_process();
    std::vector<CachelineAlignedType<std::size_t>> counts(num_threads, 0ull);
    non_init_vector<VId> vertices_to_query_local_id;
#pragma omp parallel
    {
      const auto thread_id = omp_get_thread_num();
#pragma omp for schedule(static)
      for (std::size_t i = 0; i < min_edge_idxs.size(); ++i) {
        const LocalEdgeId idx = min_edge_idxs[i];
        if (is_defined(idx)) {
          const VId& dst_id = graph.edges()[idx].get_dst();
          if (is_representative[i] == 1) {
            predecessors[i] = ComputeRepresentative::set_msb(
                predecessors[i]); // roots point to themselves
          } else {
            predecessors[i] = dst_id;
            ++counts[thread_id];
          }
        }
      }
#pragma omp single
{
      std::partial_sum(counts.begin(), counts.end(), counts.begin());
      vertices_to_query_local_id.resize(counts.back());
}

#pragma omp for schedule(static)
      for (std::size_t i = 0; i < min_edge_idxs.size(); ++i) {
        const LocalEdgeId idx = min_edge_idxs[i];
        if (is_defined(idx) && is_representative[i] != 1) {
          --counts[thread_id];
          vertices_to_query_local_id[counts[thread_id]] = i;
        }
      }
    }
    return vertices_to_query_local_id;
  }

  template <typename Graph>
  static non_init_vector<VId>
  compute_representatives(const Graph& graph, Representatives& representatives,
                          std::size_t global_round) {
    get_timer().start_phase("representative_computation");
    asm volatile("" ::: "memory");
    asm volatile("" ::: "memory");
    {
      get_timer().start("compute_reps_consec_vertices", global_round);
      mpi::MPIContext ctx;
      get_timer().stop("compute_reps_consec_vertices", global_round);

      get_timer().start("compute_reps_init", global_round);
      asm volatile("" ::: "memory");
      auto vertices_to_query_local_id =
          representatives.compute_vertices_with_nonfinal_representatives();
      asm volatile("" ::: "memory");
      get_timer().stop("compute_reps_init", global_round);

      get_timer().start("compute_reps_jumping_complete", global_round);
      asm volatile("" ::: "memory");
      const std::size_t measurement_offset = global_round * 20;

      std::size_t num_remaining_elems = vertices_to_query_local_id.size();
      parlay::sequence<VId> vertices_to_query_local_id_tmp(num_remaining_elems);
      VId* ptr_v_to_query = vertices_to_query_local_id.data();
      VId* ptr_v_to_query_tmp = vertices_to_query_local_id_tmp.data();
      int round = 0;
      auto& predecessors = representatives.get_predecessors();
      auto& home_of_predecessors = representatives.get_home_of_predecessors();
      while (pointer_jumping_round_two_level(
          graph, Span(ptr_v_to_query, num_remaining_elems), predecessors,
          home_of_predecessors, measurement_offset + round)) {
        // get_timer().add("compute_reps_jumping_num_queries",
        //                 measurement_offset + round, num_remaining_elems,
        //                 {Timer::DatapointsOperation::ID});
        asm volatile("" ::: "memory");
        get_timer().start("reps_jumping_remove", measurement_offset + round);
        asm volatile("" ::: "memory");
        parlay::slice in(ptr_v_to_query, ptr_v_to_query + num_remaining_elems);
        parlay::slice out(ptr_v_to_query_tmp,
                          ptr_v_to_query_tmp + num_remaining_elems);
        num_remaining_elems =
            parlay::filter_into(in, out, [&](const VId local_id) {
              const VId pred = predecessors[local_id];
              return !ComputeRepresentative::is_msb_set(pred);
            });
        std::swap(ptr_v_to_query, ptr_v_to_query_tmp);
        asm volatile("" ::: "memory");
        get_timer().stop("reps_jumping_remove", measurement_offset + round);
        asm volatile("" ::: "memory");
        ++round;
      }
      asm volatile("" ::: "memory");
      get_timer().stop("compute_reps_jumping_complete", global_round);

      get_timer().start("compute_reps_jumping_remove", global_round);
      asm volatile("" ::: "memory");
      representatives.reset_marks();
      asm volatile("" ::: "memory");
      get_timer().stop("compute_reps_jumping_remove", global_round);
      asm volatile("" ::: "memory");
    }
    get_timer().stop_phase();
    return representatives.extract_predecessors();
  }
}; // namespace hybridMST
struct UpdateParentArray {
  template <typename Graph>
  static void execute(const Graph& graph, ParentArray& parent_array,
                      const non_init_vector<VId>& local_vertex_new_label) {
    using VertexParent = ParentArray::VertexParent;
    non_init_vector<VertexParent> vertex_parents(local_vertex_new_label.size());
    for (std::size_t i = 0; i < local_vertex_new_label.size(); ++i) {
      const VId v = graph.get_global_id(i);
      const VId parent_v = local_vertex_new_label[i];
      vertex_parents[i] = VertexParent{v, parent_v};
    }
    parent_array.update(vertex_parents);
  }
};
} // namespace hybridMST
