#pragma once

#include <vector>
#include <unordered_map>

#include "definitions.hpp"

namespace hybridMST {
class UnionFind {
public:
  UnionFind(std::size_t n) : data(n, -1) {}
  void unify(VId a, VId b) {
    VId root_a = find(a);
    VId root_b = find(b);
    if (root_a == root_b)
      return;
    if (data[root_b] < data[root_a])
      std::swap(root_b, root_a); // sizes are encoded as negative integers
    data[root_a] += data[root_b];
    data[root_b] = root_a;
  }
  VId find(VId a) {
    if (data[a] < 0)
      return a;
    VId parent = static_cast<VId>(data[a]);
    data[a] = find(parent);
    return data[a];
  }

private:
  std::vector<std::int64_t> data;
};

class MapBasedUnionFind {
 public:
  using Map = std::unordered_map<VId, std::int64_t>;
  MapBasedUnionFind() {
    //data.set_empty_key(std::numeric_limits<std::int64_t>::max());
  }
  MapBasedUnionFind(uint/*approx_size*/) /*: data(approx_size)*/ {
    //data.set_empty_key(std::numeric_limits<std::int64_t>::max());
  }
  void unify(VId a, VId b) {
    VId root_a = find(a);
    VId root_b = find(b);
    if (root_a == root_b) return;
    if (data[root_b] < data[root_a])
      std::swap(root_b, root_a);  // sizes are encoded as negative integers
    data[root_a] += data[root_b];
    data[root_b] = root_a;
  }
  VId find(VId a) {
    const auto it = data.find(a);
    if (it == data.end()) {
      data[a] = -1;
      return a;
    }
    if (it->second < 0) return a;
    VId parent = static_cast<VId>(it->second);
    it->second = find(parent);
    return it->second;
  }

 private:
  Map data;
};
} // namespace hybridMST
