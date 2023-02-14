#pragma once

#include <algorithm>
#include <datastructures/concurrent_lookup_map.hpp>
#include <datastructures/distributed_parent_array.hpp>
#include <execution>
#include <random>

#include "RQuick/RQuick.hpp"

#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/context.hpp"
#include "mpi/scan.hpp"
#include "mpi/type_handling.hpp"
#include "util/macros.hpp"
#include "util/utils.hpp"

namespace hybridMST {
//inline Weight select_pivot_weight(const WEdgeList& edges, VertexRange range) {
//  mpi::MPIContext ctx;
//  std::mt19937 gen(ctx.rank());
//  std::uniform_int_distribution<> sample_idx(0, edges.size() - 1);
//  const std::size_t sample_size = std::min(edges.size() / 2ull, 100ull);
//  std::vector<Weight> samples(sample_size);
//  for (auto& sample : samples) {
//    sample = weight(edges[sample_idx(gen)]);
//  }
//  mpi::TypeMapper<Weight> tm;
//  int tag = 100000;
//  std::mt19937_64 gen_sort(ctx.rank());
//  RQuick::sort(tm.get_mpi_datatype(), samples, tag, gen_sort,
//               ctx.communicator());
//  const std::size_t n = mpi::allreduce_max(range.second) + 1;
//  const std::size_t m = mpi::allreduce_sum(edges.size());
//  const std::size_t num_samples = mpi::allreduce_sum(samples.size());
//  const double ratio = std::min(static_cast<double>(5 * n) / (2 * m), 1.0);
//  const std::size_t global_sample_start =
//      mpi::exscan_sum(samples.size(), ctx, 0ul);
//  const std::size_t pivot_idx = num_samples * ratio;
//  MPI_ASSERT_(pivot_idx < num_samples, "");
//  const bool selected_sample_in_local_range =
//      global_sample_start <= pivot_idx &&
//      pivot_idx < global_sample_start + samples.size();
//  Weight pivot_weight = WEIGHT_INF;
//  if (selected_sample_in_local_range) {
//    const std::size_t local_idx = pivot_idx - global_sample_start;
//    pivot_weight = samples[local_idx];
//    SEQ_EX(ctx, PRINT_VAR(pivot_weight););
//    pivot_weight = mpi::allreduce_min(pivot_weight);
//  } else {
//    SEQ_EX(ctx, PRINT_VAR(pivot_weight););
//    pivot_weight = mpi::allreduce_min(pivot_weight);
//  }
//  SEQ_EX(ctx, PRINT_VAR(pivot_weight););
//  return pivot_weight;
//}
//
//template <typename EdgeType>
//Span<EdgeType> partition_edges(Span<EdgeType> edges, const Weight pivot) {
//  auto it =
//      std::partition(std::execution::par, edges.begin(), edges.end(),
//                     [&](const auto& edge) { return weight_ref(edge) <= pivot; }
//
//      );
//  const auto size_first_partition =
//      static_cast<std::size_t>(std::distance(edges.begin(), it));
//
//  // std::cout << "pivot: " << pivot
//  //           << " partition.size(): " << size_first_partition
//  //           << " container.size(): " << container.size() << std::endl;
//  return Span(edges.data(), size_first_partition);
//}
//
//template <typename EdgeType>
//void rename_sort_edges(Span<EdgeType> edges, const ParentArray& parrent_array) {
//  mpi::MPIContext ctx;
//  parlay::hashtable<parlay::hash_numeric<VId>> table(
//      2 * edges.size(), parlay::hash_numeric<VId>{});
//#pragma omp parallel for
//  for (std::size_t i = 0; i < edges.size(); ++i) {
//    const auto& edge = edges[i];
//    const VId& src = src_ref(edge);
//    const VId& dst = dst_ref(edge);
//    table.insert(src);
//    table.insert(dst);
//  }
//  auto entries = table.entries();
//  ParallelWriteLookUpMap<VId, VId> vertex_parent_map{
//      ctx.threads_per_mpi_process()};
//  parrent_array.get_parents(entries, vertex_parent_map);
//#pragma omp parallel for
//  for (std::size_t i = 0; i < edges.size(); ++i) {
//    auto& edge = edges[i];
//    VId& src = src_ref(edge);
//    VId& dst = dst_ref(edge);
//    VId src_parent = vertex_parent_map.get(src);
//    VId dst_parent = vertex_parent_map.get(dst);
//    src = src_parent;
//    dst = dst_parent;
//  }
//  std::remove_if(
//      std::execution::par, edges.begin(), edges.end(),
//      [](const EdgeType& edge) { return src_ref(edge) == dst_ref(edge); });
//}
} // namespace hybridMST
