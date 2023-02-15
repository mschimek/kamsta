#pragma once

#include "definitions.hpp"
#include "shared_mem_parallel.hpp"

namespace hybridMST {
template <typename EdgeIds, typename Edges, typename Parents>
void relabel_edges(std::size_t m, const EdgeIds& edge_ids, Edges& edges,
                   const Parents& parents) {
  hybridMST::parallel_for(0, m, [&](size_t i) {
    size_t e_id = edge_ids[i];
    auto& edge = edges[e_id];
    const VId u = static_cast<VId>(edge.get_src());
    const VId v = static_cast<VId>(edge.get_dst());
    const VId pu = parents[u];
    const VId pv = parents[v];
    if (u != pu || v != pv) {
      edge.set_src(pu);
      edge.set_dst(pv);
    }
    if (pu == pv) {
      edge_ids[i] |=
          MSD<LocalEdgeId>; // mark self loops, so that they can be filtered out
     // std::cout << "\ti: " << i << " self: loop " << edge << " " << edge_ids[i] << std::endl;
    } else {
      //std::cout << "\ti: " << i << " not-self: loop " << edge << " " << edge_ids[i] << std::endl;
    }
  });
}

template <typename MinEdges, typename Edges, typename Vertices,
          typename Parents, typename RootsInfo, typename ExhaustionInfo,
          typename MstEdges>
void determine_mst_edges(std::size_t n, const MinEdges& min_edges,
                         const Edges& edges, const Vertices& vertices,
                         Parents& parents, RootsInfo& is_root,
                         ExhaustionInfo& exhausted, MstEdges& new_mst_edges) {
  hybridMST::parallel_for(0, n, [&](size_t i) {
    const VId v = vertices[i];
    const EdgeIdWeight& e = min_edges[v].load();
    if (e.edge_id == LOCAL_EDGEID_UNDEFINED) {
      // no more edges incident to v in this batch.
      exhausted[i] = true;
      is_root[i] = false;
      new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
    } else {
      const LocalEdgeId ind = e.edge_id;
      const auto& edge = edges[ind];
      const VId u = static_cast<VId>(edge.get_src()) ^
                    static_cast<VId>(edge.get_dst()) ^ v;
      // pick the lower endpoint as the root.
      if (u > v && ind == min_edges[u].load().edge_id) {
        parents[v] = v;
        is_root[i] = true;
        new_mst_edges[i] = LOCAL_EDGEID_UNDEFINED;
      } else {
        // v is satellite: hook onto u.
        parents[v] = u;
        is_root[i] = false;
        new_mst_edges[i] = ind;
      }
    }
  });
}

} // namespace hybridMST
