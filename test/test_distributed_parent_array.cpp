#include <random>
#include <unordered_set>

#include "catch2/catch.hpp"

#include "datastructures/distributed_parent_array.hpp"

namespace hybridMST::tests {
template <typename Container>
std::vector<typename Container::value_type>
make_vector(const Container& container) {
  using T = typename Container::value_type;
  std::vector<T> vector;
  for (const auto& elem : container) {
    vector.push_back(elem);
  }
  return vector;
}
std::unordered_set<VId> get_roots(std::vector<VId>& vertices,
                                  double frac_of_roots) {
  if (vertices.empty())
    return {};
  std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> idx(0, vertices.size() - 1);
  const std::size_t num_roots = vertices.size() * frac_of_roots;
  std::unordered_set<VId> roots;
  while (roots.size() < num_roots) {
    roots.insert(vertices[idx(gen)]);
  }
  return roots;
}
std::unordered_set<VId> get_roots(std::unordered_set<VId>& vertices,
                                  double frac_of_roots) {
  auto vertices_vec = make_vector(vertices);
  return get_roots(vertices_vec, frac_of_roots);
}

std::unordered_map<VId, VId>
compute_updates(std::unordered_set<VId>& active_vertices,
                std::unordered_set<VId>& roots) {
  if (roots.empty())
    return {};
  std::unordered_map<VId, VId> new_parent;
  auto roots_vec = make_vector(roots);
  std::mt19937 gen(0);
  std::uniform_int_distribution<> new_root(0, roots_vec.size() - 1);
  for (const auto& active_vertex : active_vertices) {
    if(roots.find(active_vertex) != roots.end())
      continue; // do not update root
    new_parent[active_vertex] = roots_vec[new_root(gen)];
  }
  return new_parent;
}
void shortcut(std::vector<VId>& parents) {
  for (std::size_t i = 0; i < parents.size(); ++i) {
    while (parents[i] != parents[parents[i]])
      parents[i] = parents[parents[i]];
  }
}
void simulate_boruvka(std::size_t n, std::size_t num_components) {
  mpi::MPIContext ctx;
  ParentArray parent_array(n - 1);
  std::vector<VId> parents(n);
  const VId v_begin = n / ctx.size() * ctx.rank();
  const VId v_end = (n / ctx.size() * (ctx.rank() + 1)) +
                    ((ctx.rank() + 1 == ctx.size()) ? n % ctx.size() : 0);
  std::iota(parents.begin(), parents.end(), 0ull);
  const double frac_of_roots = 0.75;
  std::unordered_set<VId> active_vertices;
  for (const VId& v : parents) {
    active_vertices.insert(v);
  }
  while (active_vertices.size() > num_components) {
    auto new_roots = get_roots(active_vertices, frac_of_roots);
    if (ctx.rank() == 0) {
      PRINT_CONTAINER_WITH_INDEX(parents);
      PRINT_SET(active_vertices);
      PRINT_SET(new_roots);
    }
    // SEQ_EX(ctx, PRINT_VAR(parent_array););
    std::vector<ParentArray::VertexParent> v_new_parent;
    for (const auto& [v, parent] :
         compute_updates(active_vertices, new_roots)) {
      parents[v] = parent;
      if (v_begin <= v && v < v_end) {
        v_new_parent.emplace_back(v, parent);
      }
      parent_array.update(v_new_parent);
      for (std::size_t i = parent_array.v_begin(); i < parent_array.v_end();
           ++i) {
        REQUIRE(parent_array.get_parent(i) == parents[i]);
      }
      active_vertices = new_roots;
    }
  }
  if (ctx.rank() == 0) {
    PRINT_SET(active_vertices);
    PRINT_CONTAINER_WITH_INDEX(parents);
  }
  // SEQ_EX(ctx, PRINT_VAR(parent_array););
  if (ctx.rank() == 0)
    std::cout << " start shortcut " << std::endl;
  shortcut(parents);
  if (ctx.rank() == 0)
    std::cout << " start distributed shortcut " << std::endl;
  parent_array.shortcut();
  // if (ctx.rank() == 0) {
  //   PRINT_CONTAINER_WITH_INDEX(parents);
  // }
  // SEQ_EX(ctx, PRINT_VAR(parent_array););
  for (VId v = parent_array.v_begin(); v < parent_array.v_end(); ++v) {
    REQUIRE(parent_array.get_parent(v) == parents[v]);
  }
}
} // namespace hybridMST::tests

TEST_CASE("Distributed Parent Array 0", "[utils]") {
  hybridMST::tests::simulate_boruvka(15, 3);
}

TEST_CASE("Distributed Parent Array 1", "[utils]") {
  hybridMST::tests::simulate_boruvka(39, 3);
}
TEST_CASE("Distributed Parent Array 2", "[utils]") {
  hybridMST::tests::simulate_boruvka(99, 1);
}
TEST_CASE("Distributed Parent Array 3", "[utils]") {
  hybridMST::tests::simulate_boruvka(533, 10);
}
TEST_CASE("Distributed Parent Array 4", "[utils]") {
  hybridMST::tests::simulate_boruvka(533, 533);
}
TEST_CASE("Distributed Parent Array 5", "[utils]") {
  hybridMST::tests::simulate_boruvka(1024, 533);
}
