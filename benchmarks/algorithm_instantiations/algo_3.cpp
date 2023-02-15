#include "algorithms/hybrid_boruvka.hpp"
#include "definitions.hpp"

template hybridMST::WEdgeList
hybridMST::boruvka(std::vector<hybridMST::WEdge>,
                   const hybridMST::AlgorithmConfig<hybridMST::WEdge_6_1,
                                                    hybridMST::WEdgeId_6_1_7>&);
//extern template hybridMST::WEdgeList hybridMST::boruvka(
//    std::vector<hybridMST::WEdge_6_1>,
//    const hybridMST::AlgorithmConfig<hybridMST::WEdge, hybridMST::WEdgeId>&);
