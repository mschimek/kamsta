#include "algorithms/hybrid_boruvka.hpp"
#include "definitions.hpp"

template hybridMST::WEdgeList hybridMST::boruvka(
    std::vector<hybridMST::WEdge>,
    const hybridMST::AlgorithmConfig<hybridMST::WEdge, hybridMST::WEdgeId>&);
