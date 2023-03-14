#pragma once

#include <numeric>

#include "datastructures/algorithms/pointer_jumping_on_dist_array.hpp"
#include "datastructures/distributed_array.hpp"
#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/context.hpp"
#include "util/macros.hpp"
#include "util/utils.hpp"

namespace hybridMST {

class ParentArray {
public:
  struct InHashMap {};
  struct InVector {};
  using VertexParent = typename DistributedArray<VId>::IndexValue;
  using VertexHasEdges = typename DistributedArray<std::uint8_t>::IndexValue;
  ParentArray(std::size_t n)
      : parents_(n,
                 [&](std::size_t i, std::size_t begin) { return i + begin; }),
        has_edges_initially_(n, [&](std::size_t /*i*/, std::size_t /*begin*/) {
          return false;
        }) {}

  template <typename Container>
  void update(const Container& orgId_new_parentIds) {
    parents_.set_values(orgId_new_parentIds);
  }
  template <typename Container>
  void set_non_isolated_vertices(const Container& non_isolated_vertices) {
    has_edges_initially_.set_values(non_isolated_vertices);
  }
  template <typename Vertices>
  auto get_parents(const Vertices& vertices,
                   InHashMap default_tag = InHashMap{}) const {
    return parents_.get_values(vertices);
  }
  template <typename Vertices>
  auto get_parents(const Vertices& vertices, InVector /*tag*/) const {
    return parents_.get_values_in_vector_filter(vertices);
  }
  const VId& get_parent(const VId& v) const {
    return parents_.get_value_locally(v);
  }
  VId v_begin() const { return parents_.index_begin(); }
  VId v_end() const { return parents_.index_end(); }

  void shortcut() {
    const auto is_root = [](VId v_global, VId v_parent) {
      return v_global == v_parent;
    };
    ParallelPointerJumping<VId>::execute(parents_, is_root);
  }

  VId global_num_vertices() const {
    const VId begin = v_begin();
    const VId end = v_end();
    std::size_t num_roots = 0;
#pragma omp parallel for reduction(+ : num_roots)
    for (std::size_t i = begin; i < end; ++i) {
      num_roots +=
          ((get_parent(i) == i) && has_edges_initially_.get_value_locally(i));
    }
    return mpi::allreduce_sum(num_roots);
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const ParentArray& parent_array) {
    out << parent_array.parents_;
    return out;
  }

private:
  const mpi::MPIContext ctx_;
  DistributedArray<VId> parents_;
  DistributedArray<std::uint8_t> has_edges_initially_;
};
} // namespace hybridMST
