#pragma once

#include "datastructures/variable_length_encoding.hpp"
#include "definitions.hpp"
#include "ips4o/ips4o.hpp"
#include "mpi/context.hpp"
#include "mpi/scan.hpp"
#include "tlx/math/clz.hpp"
#include <util/utils.hpp>

namespace hybridMST {
template <typename WEdgeType_, typename WEdgeIdType_>
class DifferenceEncodedGraph {
public:
  using WEdgeType = WEdgeType_;
  using WEdgeIdType = WEdgeIdType_;
  using WEdgeContainer = non_init_vector<WEdgeType>;
  using WEdgeIdContainer = non_init_vector<WEdgeIdType>;

  template <typename Edges>
  DifferenceEncodedGraph(const Edges& input_edges, std::size_t num_threads,
                         std::size_t edge_index_offset, VertexRange range)
      : num_edges{input_edges.size()}, num_threads{num_threads},
        chunk_length{num_edges / num_threads},
        edge_index_offset{edge_index_offset}, range{range} {
    mpi::MPIContext ctx;
    for (std::size_t i = 0; i < num_threads; ++i) {
      const auto& input_edge = input_edges[get_begin_index(i)];
      WEdgeType edge;
      edge.set_src(input_edge.get_src());
      edge.set_dst(input_edge.get_dst());
      edge.set_weight(input_edge.get_weight());
      start_edge.push_back(edge);
    }
    start_index_data.resize(num_threads + 1, 0ull); // sentinel element;
    const std::size_t num_required_bytes =
        precompute_space_consumption(input_edges);
    data.resize(num_required_bytes);
    encode_edges(input_edges);
    if (ctx.rank() == 0) {
      std::cout << "finished encoding: " << (data.size()) << std::endl;
      std::cout << "finished encoding: " << (input_edges.size() * sizeof(WEdge))
                << std::endl;
    }
  }

  WEdgeList get_WEdgeList() const {
    WEdgeList edges(num_edges);
#pragma omp parallel for
    for (std::size_t task = 0; task < num_threads; ++task) {
      auto it = data.data() + start_index_data[task];
      auto prev_src = start_edge[task].get_src();
      WEdge edge{start_edge[task].get_src(), start_edge[task].get_dst(),
                 start_edge[task].get_weight()};
      edges[task * chunk_length] = edge;
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = decode_edge(prev_src, it);
        edges[i] = WEdge(edge.get_src(), edge.get_dst(), edge.get_weight());
        prev_src = edge.get_src();
      }
    }
    return edges;
  }

  WEdgeContainer get_WEdges() const {
    WEdgeContainer edges(num_edges);
#pragma omp parallel for
    for (std::size_t task = 0; task < num_threads; ++task) {
      auto it = data.data() + start_index_data[task];
      auto prev_src = start_edge[task].get_src();
      edges[task * chunk_length] = start_edge[task];
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = decode_edge(prev_src, it);
        edges[i] = edge;
        prev_src = edge.get_src();
      }
    }
    return edges;
  }

  WEdgeIdContainer get_WEdgeIds() const {
    WEdgeIdContainer edges(num_edges);
#pragma omp parallel for
    for (std::size_t task = 0; task < num_threads; ++task) {
      auto it = data.data() + start_index_data[task];
      const auto& start_edge_ = start_edge[task];
      WEdgeIdType start_edge;
      start_edge.set_src(start_edge_.get_src());
      start_edge.set_dst(start_edge_.get_dst());
      start_edge.set_weight(start_edge_.get_weight());
      start_edge.set_edge_id(get_begin_index(task) + edge_index_offset);
      edges[get_begin_index(task)] = start_edge;
      VId prev_src = start_edge.get_src();
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = decode_edge(prev_src, it, i + edge_index_offset);
        edges[i] = edge;
        prev_src = edge.get_src();
      }
    }
    return edges;
  }
  template <typename Indices>
  WEdgeContainer get_WEdges(const Indices& local_indices) {
    WEdgeContainer edges(local_indices.size());
    ips4o::parallel::sort(local_indices.begin(), local_indices.end());
#pragma omp parallel for
    for (std::size_t task = 0; task < num_threads; ++task) {
      auto it = data.data() + start_index_data[task];
      auto edge = start_edge[task];
      edges[get_begin_index(task)] = edge;
      VId prev_src = edge.get_src();
      auto it_in_indices = std::lower_bound(
          local_indices.begin(), local_indices.end(), get_begin_index(task));
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = decode_edge(prev_src, it);
        for (; *it_in_indices < i && it_in_indices < local_indices.end(); ++it)
          ;
        if (it != local_indices.end() && *it == i) {
          edges[std::distance(local_indices.begin(), it)] = edge;
        }
        prev_src = edge.get_src();
      }
    }
    return edges;
  }
  VertexRange get_range() const { return range; }
  std::size_t num_local_edges() const { return num_edges; }

private:
  std::size_t get_begin_index(std::size_t task_num) const {
    return chunk_length * task_num;
  }
  std::size_t get_end_index(std::size_t task_num) const {
    std::size_t end = chunk_length * (task_num + 1);
    end += (task_num + 1 == num_threads) ? num_edges % num_threads : 0;
    return end;
  }
  template <typename InputEdges>
  void encode_edges(const InputEdges& input_edges) {
#pragma omp parallel for
    for (std::size_t task = 0; task < num_threads; ++task) {
      auto output_it = data.data() + start_index_data[task];
      WEdgeType prev_edge;
      {
        const auto& e = start_edge[task];
        prev_edge.set_src(e.get_src());
        prev_edge.set_dst(e.get_dst());
        prev_edge.set_weight(e.get_weight());
      }
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = input_edges[i];
        output_it =
            encode_value(edge.get_src() - prev_edge.get_src(), output_it);
        output_it = encode_value(edge.get_dst(), output_it);
        output_it = encode_value(edge.get_weight(), output_it);
        prev_edge.set_src(edge.get_src());
        prev_edge.set_dst(edge.get_dst());
        prev_edge.set_weight(edge.get_weight());
      }
      if (output_it - data.data() != start_index_data[task + 1]) {
        std::stringstream sstream;
        sstream << "wrong index: " << task << " " << output_it - data.data()
                << " " << start_index_data[task + 1];
        std::cout << sstream.str() << std::endl;
        std::abort();
      }
    }
  }
  template <typename It>
  WEdgeType decode_edge(const VId& prev_src, It& it) const {
    VId src = decode_value(it) + prev_src;
    VId dst = decode_value(it);
    Weight w = decode_value(it);
    WEdgeType edge;
    edge.set_src(src);
    edge.set_dst(dst);
    edge.set_weight(w);
    return edge;
  }
  template <typename It>
  WEdgeIdType decode_edge(const VId& prev_src, It& it,
                          const std::size_t index) const {
    VId src = decode_value(it) + prev_src;
    VId dst = decode_value(it);
    Weight w = decode_value(it);
    WEdgeIdType e;
    e.set_src(src);
    e.set_dst(dst);
    e.set_weight(w);
    e.set_edge_id(index);
    return e;
  }
  template<typename InputEdges>
  std::size_t precompute_space_consumption(const InputEdges& input_edges) {
    std::size_t num_required_bytes = 0;
#pragma omp parallel for reduction(+ : num_required_bytes)
    for (std::size_t task = 0; task < num_threads; ++task) {

      VId prev_src = start_edge[task].get_src();
      for (std::size_t i = get_begin_index(task) + 1; i < get_end_index(task);
           ++i) {
        const auto& edge = input_edges[i];
        num_required_bytes += num_bytes_for_encoding(edge.get_src() - prev_src);
        num_required_bytes += num_bytes_for_encoding(edge.get_dst());
        num_required_bytes += num_bytes_for_encoding(edge.get_weight());
        prev_src = edge.get_src();
      }
      start_index_data[task] = num_required_bytes;
    }
    std::exclusive_scan(start_index_data.begin(), start_index_data.end(),
                        start_index_data.begin(), 0ull);
    return num_required_bytes;
  }
  std::size_t num_bytes_for_encoding(const uint64_t& value) const {
    const std::size_t pos_of_most_sig_one =
        std::numeric_limits<uint64_t>::digits - tlx::clz(value);
    const std::size_t divisor = pos_of_most_sig_one / 7ul;
    const bool remainder = pos_of_most_sig_one % 7ul != 0;
    return std::max(divisor + remainder, static_cast<std::size_t>(1ull));
  }

  std::vector<WEdgeType> start_edge;
  std::vector<std::size_t> start_index_data;
  non_init_vector<unsigned char> data;
  std::size_t num_edges;
  std::size_t num_threads;
  std::size_t chunk_length;
  std::size_t edge_index_offset;
  VertexRange range;
};

template <typename InputEdges, typename WEdgeType, typename WEdgeIdType>
void verify_compressed_construction(const InputEdges& input_edges,
                                    const VertexRange& range) {
  mpi::MPIContext ctx;
  auto start = std::chrono::steady_clock::now();
  std::size_t edge_offset =
      hybridMST::mpi::exscan_sum(input_edges.size(), ctx, 0ul);
  hybridMST::DifferenceEncodedGraph<WEdgeType, WEdgeIdType> compressed_graph(
      input_edges, ctx.threads_per_mpi_process(), edge_offset, range);
  auto end = std::chrono::steady_clock::now();
  const auto construction_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      static_cast<double>(1000'000);
  if (ctx.rank() == 0) {
    std::cout << "construction time: " << construction_time << std::endl;
  }
  start = std::chrono::steady_clock::now();
  auto decoded_edges = compressed_graph.get_WEdges();
  end = std::chrono::steady_clock::now();
  const auto decoding_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      static_cast<double>(1000'000);
  if (ctx.rank() == 0) {
    std::cout << "decode time: " << decoding_time << std::endl;
  }
  // result verification
  const bool has_equal_size = decoded_edges.size() == input_edges.size();
  ctx.mpi_assert(has_equal_size, " size not equal");
  using InputEdgeType = typename InputEdges::value_type;
  for (std::size_t i = 0; has_equal_size && i < input_edges.size(); ++i) {
    auto comp = hybridMST::SrcDstWeightEqual<InputEdgeType, WEdgeType>{};
    if (!comp(input_edges[i], decoded_edges[i])) {
      SEQ_EX(ctx, PRINT_CONTAINER_WITH_INDEX(input_edges);
             PRINT_CONTAINER_WITH_INDEX(decoded_edges););
    }
  }
}
} // namespace hybridMST
