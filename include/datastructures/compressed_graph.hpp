#pragma once

#include <bit>

#include "parlay/primitives.h"
#include "tlx/math/clz.hpp"

#include "definitions.hpp"
#include "variable_length_encoding.hpp"

namespace hybridMST {

struct Indices {
  using IndexingType = std::uint32_t;
  Indices(std::size_t n)
      : n_{n},
        slice_size_{/*std::numeric_limits<IndexingType>::max() /
                    offset_factor*/
                    1'000'000},
        num_slices{n / slice_size_ + 1}, indices(n),
        offsets(num_slices + 5ull, 0) {}
  std::size_t get_slice(std::size_t i) const { return i / slice_size_; }
  std::size_t get_value(std::size_t i) const {
    return indices[i] + offsets[get_slice(i)];
  }
  auto begin_slice(std::size_t slice_index) {
    return indices.begin() + slice_index * slice_size_;
  }
  auto end_slice(std::size_t slice_index) {
    return indices.begin() + std::min((slice_index + 1) * slice_size_, n_);
  }
  std::size_t begin_idx_slice(std::size_t slice_index) const {
    return slice_index * slice_size_;
  }
  std::size_t end_idx_slice(std::size_t slice_index) const {
    return std::min((slice_index + 1) * slice_size_, n_);
  }
  IndexingType& operator[](std::size_t i) { return indices[i]; }
  const IndexingType& operator[](std::size_t i) const { return indices[i]; }
  void set_offset(std::size_t i, IndexingType value) { offsets[i] = value; }
  auto data() { return indices.data(); }

  const std::size_t offset_factor = 32;
  const std::size_t n_;
  const std::size_t slice_size_;
  const std::size_t num_slices;
  non_init_vector<IndexingType> indices;
  std::vector<uint64_t> offsets;
};
class CompressedLocalGraph {
public:
  CompressedLocalGraph(const WEdgeList& edges) : indices(edges.size()) {
    if (edges.size() * sizeof(WEdge) > std::numeric_limits<uint32_t>::max()) {
      std::abort();
    }
    v_min = edges.empty() ? 0 : edges.front().get_src();
    // PRINT_VAR(edges.size());
    const std::size_t num_required_bytes = precompute_space_consumption(edges);

    // PRINT_CONTAINER_WITH_INDEX(indices.indices);
    // PRINT_CONTAINER_WITH_INDEX(indices.offsets);
    //  PRINT_VAR(num_required_bytes);
    data.resize(num_required_bytes);
    encode_edges(edges);
    std::cout << "finished encoding: "
              << (data.size() + indices.indices.size() * 4) << std::endl;
    std::cout << "finished encoding: " << (edges.size() * sizeof(WEdge))
              << std::endl;
  }
  WEdgeList decode_edges() {
    WEdgeList edges(indices.n_);
#pragma omp parallel for
    for (std::size_t i = 0; i < indices.n_; ++i) {
      edges[i] = decode_edge(i);
    }
    return edges;
  }
  non_init_vector<WEdgeId> decode_edges_with_idx() {
    non_init_vector<WEdgeId> edges(indices.n_);
#pragma omp parallel for
    for (std::size_t i = 0; i < indices.n_; ++i) {
      edges[i] = decode_edge_with_idx(i);
    }
    return edges;
  }

private:
  int num_bytes_for_encoding(const uint64_t& value) const {
    const std::size_t pos_of_most_sig_one =
        std::numeric_limits<uint64_t>::digits - tlx::clz(value);
    const std::size_t divisor = pos_of_most_sig_one / 7;
    const bool remainder = pos_of_most_sig_one % 7 != 0;
    return std::max(divisor + remainder, static_cast<std::size_t>(1ull));
  }

  std::size_t precompute_space_consumption(const WEdgeList& input_edges) {
    // indices is not initialized yet
    std::size_t complete_num_required_bytes = 0;
    for (std::size_t cur_slice = 0; cur_slice < indices.num_slices;
         ++cur_slice) {
      const auto begin_idx = indices.begin_idx_slice(cur_slice);
      const auto end_idx = indices.end_idx_slice(cur_slice);
#pragma omp parallel for
      for (std::size_t i = begin_idx; i < end_idx; ++i) {
        const auto& edge = input_edges[i];
        indices[i] = 0;
        indices[i] += num_bytes_for_encoding(edge.get_src() - v_min);
        indices[i] += num_bytes_for_encoding(edge.get_dst());
        indices[i] += num_bytes_for_encoding(edge.get_weight());
      }
      auto slice = parlay::slice(indices.begin_slice(cur_slice),
                                 indices.end_slice(cur_slice));
      const std::size_t num_required_bytes = parlay::reduce(slice);
      // PRINT_CONTAINER_WITH_INDEX(indices.indices);
      parlay::scan_inplace(slice);
      indices.set_offset(cur_slice + 1, num_required_bytes);
      complete_num_required_bytes += num_required_bytes;
      // PRINT_CONTAINER_WITH_INDEX(indices.indices);
    }
    std::inclusive_scan(indices.offsets.begin(), indices.offsets.end(),
                        indices.offsets.begin());
    return complete_num_required_bytes;
  }

  void encode_edges(const WEdgeList& input_edges) {
#pragma omp parallel for
    for (std::size_t i = 0; i < input_edges.size(); ++i) {
      const auto& edge = input_edges[i];
      auto expanded_index = indices.get_value(i);
      auto output_it = data.data() + expanded_index;
      output_it = encode_value(edge.get_src() - v_min, output_it);
      output_it = encode_value(edge.get_dst(), output_it);
      output_it = encode_value(edge.get_weight(), output_it);
      if (output_it != data.data() + indices.get_value(i + 1) &&
          (i + 1 != input_edges.size())) {

        std::stringstream sstream;
        sstream << "wrong index: " << i << edge << " "
                << (output_it - data.data()) << " " << indices.get_value(i)
                << " " << indices.get_value(i + 1)
                << "num bytes: " << num_bytes_for_encoding(edge.get_src())
                << " " << num_bytes_for_encoding(edge.get_dst()) << " "
                << num_bytes_for_encoding(edge.get_weight()) << std::endl;
        std::cout << sstream.str() << std::endl;
        std::abort();
      }
    }
  }
  WEdge decode_edge(std::size_t i) {
    const std::size_t expanded_index = indices.get_value(i);
    auto it = data.data() + expanded_index;
    VId src = decode_value(it) + v_min;
    VId dst = decode_value(it);
    Weight w = decode_value(it);
    const auto edge = WEdge(src, dst, w);
    // std::cout << edge << std::endl;
    return edge;
  }
  WEdgeId decode_edge_with_idx(std::size_t i) {
    const std::size_t expanded_index = indices.get_value(i);
    auto it = data.data() + expanded_index;
    VId src = decode_value(it) + v_min;
    VId dst = decode_value(it);
    Weight w = decode_value(it);
    return WEdgeId(src, dst, w, i);
  }

  VId v_min;
  non_init_vector<unsigned char> data;
  Indices indices;
};


} // namespace hybridMST
