#include "definitions.hpp"
#include "util/classifier.hpp"
#include <numeric>
#include <omp.h>
#include <parlay/hash_table.h>
#include <random>


std::vector<int> gen_data(std::size_t n) {
  std::mt19937 gen(1);
  std::uniform_int_distribution<> distrib(0, 1000);
  std::vector<int> vec(n);
  for (std::size_t i = 0; i < n; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}
int main() {
  std::size_t num_splitters = 165;
  std::vector<int> splitters(num_splitters);
  for (std::size_t i = 0; i < num_splitters; ++i) {
    splitters[i] = (i + 1) * 10;
  }
  Classifier_<8, std::vector<int>> classifier_(
      splitters, std::numeric_limits<int>::max());
  Classifier classifier(splitters, std::numeric_limits<int>::max());
  const std::size_t n = 200'000'000;
  std::vector<std::size_t> res1(n);
  std::vector<std::size_t> res2(n);
  std::vector<std::size_t> res3(n);
  std::vector<std::size_t> res4(n);
  std::vector<std::size_t> res5(n);
  const auto data = gen_data(n);
  asm volatile("" ::: "memory");
  auto t1 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
  for (std::size_t i = 0; i < n; ++i) {
    res1[i] = classifier_.compute_bucket(data[i]);
  }
  asm volatile("" ::: "memory");
  auto t2 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
  for (std::size_t i = 0; i < n; ++i) {
    const auto it =
        std::lower_bound(splitters.begin(), splitters.end(), data[i]);
    res2[i] = std::distance(splitters.begin(), it);
  }
  asm volatile("" ::: "memory");
  auto t3 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
  for (std::size_t i = 0; i < n; ++i) {
    res3[i] = classifier.compute_bucket(data[i]);
  }
  asm volatile("" ::: "memory");
  auto t4 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    if (res1[i] != 0)
      res4[i] = res1[i];
  }
  asm volatile("" ::: "memory");
  auto t5 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
  omp_set_num_threads(2);

  auto yield = [&](const std::size_t& i, const std::size_t& bucket_idx) {
    res5[i] = bucket_idx;
  };
  const std::size_t log_splitter = std::ceil(std::log2(splitters.size()));
  std::cout << log_splitter << std::endl;
  auto res55 = classify(log_splitter, splitters, std::numeric_limits<int>::max(), data.begin(), data.end(), yield);
  asm volatile("" ::: "memory");
  auto t6 = std::chrono::steady_clock::now();
  asm volatile("" ::: "memory");
  std::size_t counter = 0;
  for (std::size_t i = 0; i < n; ++i) {
    counter += res1[i] == ((res2[i] == res3[i]) == res4[i]) == res55[i];
  }
  auto ts1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  auto ts2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2);
  auto ts3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);
  auto ts4 = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4);
  auto ts5 = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5);

  std::cout << "counter: " << counter << " classifier_: " << ts1.count()
            << " binary_search: " << ts2.count()
            << " classifier: " << ts3.count() << " copy: " << ts4.count()
            << " parallel: " << ts5.count() << std::endl;
}
