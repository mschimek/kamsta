#pragma once

#include "definitions.hpp"

namespace hybridMST {
enum class LocalPreprocessing {
  NO_PREPROCESSING = 0, /// no preprocessing is performed
  MODIFIED_BORUVKA =
      1, /// we run a modified version of Boruvka's algorithm on the local edges
         /// that stops contraction if there is a lighter incident cut edge
  REDUCE_EDGES_THEN_BORUVKA =
      2 /// we first identify the possible MST edges among local and cut edges
        /// and the run MODIFIED_BORUVKA on these edges.
        /// This guarantees a remaining number of local edges <= |V_i| on PE i.
        /// TODO implement this
};

enum class Compression {
  NO_COMPRESSION = 0, /// no compression is performed
  SEVEN_BIT_DIFF_ENCODING = 1
  /// We store the differences between consecutive edges e = (src, dst, w), e' =
  /// (src', dst', w') as SEVEN_BIT(src, dst, w) SEVEN_BIT(src' - src, dst',
  /// w') assuming that e is the first edge and e' its successor
  /// using 7-bit variable length encoding
};

template <typename WEdgeType_ = WEdge, typename WEdgeIdType_ = WEdgeId>
struct AlgorithmConfig {
  using WEdgeType = WEdgeType_;
  using WEdgeIdType = WEdgeIdType_;
  LocalPreprocessing local_preprocessing = LocalPreprocessing::NO_PREPROCESSING;
  Compression compression = Compression::NO_COMPRESSION;
  std::size_t filter_threshold = 4;
};

} // namespace hybridMST
