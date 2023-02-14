#pragma once

#include <iomanip>
#include <limits>
#include <random>
#include <type_traits>

#include "util/allocators.hpp"
#include "util/benchmark_helpers.hpp"
#include "util/macros.hpp"

namespace hybridMST::benchmarks {

enum class Algorithm { Sparse, Dense, TwoLevel };

inline EnumMapper<Algorithm> algorithms{
    std::make_pair(Algorithm::Sparse, std::string("Sparse")),
    std::make_pair(Algorithm::Dense, std::string("Dense")),
    std::make_pair(Algorithm::TwoLevel, std::string("TwoLevel"))};

struct AllToAllParams {
  std::size_t log_n = 10;
  double density = 1.0;
  double variance = 0.0;
  std::string algo = "Dense";
};

struct CmdParameters {
  AllToAllParams alltoall_params;
  std::size_t iterations = 1;
  bool do_check = false;
  std::size_t debug_level = 0;
  std::size_t threads_per_mpi_process = 1;
  friend std::ostream& operator<<(std::ostream& out,
                                  const CmdParameters& parameters) {
    out << " log_n=" << parameters.alltoall_params.log_n;
    out << " density=" << parameters.alltoall_params.density;
    out << " variance=" << parameters.alltoall_params.variance;
    out << " algo=" << parameters.alltoall_params.algo;
    out << " threads_per_mpi_proc=" << parameters.threads_per_mpi_process;
    out << " nb_iterations=" << parameters.iterations;
    return out;
  }
};
template <typename T = std::int32_t> class AllToAllData {
public:
  struct Message {
    int dst_pe = -1;
    std::size_t begin = 0;
    std::size_t size = 0;
    friend std::ostream& operator<<(std::ostream& out, const Message& msg) {
      return out << "(" << std::setw(10) << msg.dst_pe << ", " << std::setw(10)
                 << msg.begin << ", " << std::setw(10) << msg.size << ")";
    }
  };
  AllToAllData(std::size_t size_per_pe, double density,
               const hybridMST::mpi::MPIContext& ctx)
      : ctx(ctx), gen(ctx.rank()) {
    static_assert(std::is_integral_v<T>);

    const auto virtual_comm_partners = get_virtual_comm_partner(density);
    const std::size_t chunk_size = size_per_pe / ctx.total_size();
    generate_send_data(virtual_comm_partners.size() * chunk_size);
    transform_virtual_messages_to_real(virtual_comm_partners, chunk_size);
  }
  int get_dst(std::size_t idx) const {
    auto it =
        std::upper_bound(messages.begin(), messages.end(), idx,
                         [](const std::size_t& value, const Message& msg) {
                           return value < msg.begin;
                         });
    return std::prev(it, 1)->dst_pe;
  }
  const std::vector<T>& get_data() const { return data; }
  std::vector<T>& get_data() { return data; }
  const std::vector<Message>& get_messages() const { return messages; }

private:
  int real_comm_partner(int id, const hybridMST::mpi::MPIContext& ctx) const {
    return id / ctx.threads_per_mpi_process();
  }
  void generate_send_data(std::size_t size) {
    const auto max_value = std::numeric_limits<T>::max() - 10;
    std::uniform_int_distribution<T> value_gen(0, max_value);
    data.resize(size);
#pragma omp parallel for schedule(dynamic) firstprivate(gen, value_gen)
    for (std::size_t i = 0; i < size; ++i) {
      if (i % 50 == 0) {
        const auto thread_id = omp_get_thread_num();
        gen.discard(thread_id);
      }
      data[i] = value_gen(gen);
    }
  }
  std::vector<int> get_virtual_comm_partner(double density) {
    std::vector<int> comm_partners(ctx.total_size());
    std::iota(comm_partners.begin(), comm_partners.end(), 0);
    std::shuffle(comm_partners.begin(), comm_partners.end(), gen);
    const int nb_comm_partner = ctx.total_size() * std::min(density, 1.0);
    comm_partners.resize(nb_comm_partner);

    return comm_partners;
  }
  void transform_virtual_messages_to_real(
      const std::vector<int>& virtual_comm_partner,
      const std::size_t chunk_size) {
    std::map<int, std::size_t> commPartner_size;
    // SEQ_EX(ctx, PRINT_VECTOR(virtual_comm_partner););
    for (const auto& comm_partner : virtual_comm_partner) {
      const auto real_partner = real_comm_partner(comm_partner, ctx);
      commPartner_size[real_partner] += chunk_size;
    }
    messages.clear();
    // SEQ_EX(ctx, PRINT_VAR(commPartner_size.size()););
    for (const auto& [real_partner, size] : commPartner_size) {
      messages.push_back(Message{real_partner, 0, size});
    }
    // SEQ_EX(ctx, PRINT_VECTOR(messages););
    std::shuffle(messages.begin(), messages.end(), gen);
    // SEQ_EX(ctx, PRINT_VECTOR(messages););
    for (std::size_t i = 1; i < messages.size(); ++i) {
      const auto& prev = messages[i - 1];
      messages[i].begin = prev.begin + prev.size;
    }
  }

public:
  const hybridMST::mpi::MPIContext& ctx;
  std::mt19937 gen;
  std::vector<Message> messages;
  std::vector<T> data;
};
inline AllToAllData<int32_t> generate_data(const CmdParameters& params) {
  const hybridMST::mpi::MPIContext ctx;
  const auto log_n = params.alltoall_params.log_n;
  const std::size_t size_per_pe = (1ull << log_n) / ctx.size();
  auto res =
      AllToAllData<int32_t>{size_per_pe, params.alltoall_params.density, ctx};
  SEQ_EX(ctx, PRINT_VECTOR(res.messages););
  return res;
}
} // namespace hybridMST::benchmarks
