#pragma once

#include <algorithm>
#include <cmath>

#include "definitions.hpp"
#include "util/macros.hpp"

template <typename Container,
          typename Comparator = std::less<typename Container::value_type>>
class Classifier {
public:
  template <typename T>
  Classifier(Container splitters, const T inf_elem, Comparator&& comp)
      : comp_{std::forward<Comparator>(comp)},
        log_num_buckets{std::ceil(std::log2(splitters.size()))},
        num_buckets{1ull << log_num_buckets} {
    splitters_.resize(num_buckets + 1,
                      inf_elem); // since we start indexing at 1
    std::sort(splitters.begin(), splitters.end(), comp_);
    splitters_.resize(num_buckets, inf_elem);
    for (std::size_t i = 1; i <= log_num_buckets; ++i) {
      std::size_t n = num_buckets / (1ull << i);
      const auto begin = splitters_.begin() + n;
      const auto end = begin + n;
      // PRINT_VAR(std::distance(splitters_.begin(), begin));
      // PRINT_VAR(std::distance(splitters_.begin(), end));
      std::generate(begin, end, [&, base = 1ull << (i - 1)]() mutable {
        const auto res =
            (base - 1) < splitters.size() ? splitters[base - 1] : inf_elem;
        base += 1ull << i;
        return res;
      });
    }
  }

  template <typename T> std::size_t compute_bucket(const T& elem) const {
    std::size_t bucket_idx = 1;
    for (std::size_t i = 0; i < log_num_buckets; ++i) {
      bucket_idx = 2 * bucket_idx + (comp(splitters_[bucket_idx], elem));
    }
    bucket_idx = bucket_idx - num_buckets;
    return bucket_idx;
  }

private:
  Container splitters_;
  Comparator comp_;
  std::size_t log_num_buckets;
  std::size_t num_buckets;
};

template <std::size_t log_num_buckets, typename Container> class Classifier_ {
public:
  using T = typename Container::value_type;
  Classifier_(Container splitters, const T inf_elem) {
    splitters_.resize(num_buckets + 1,
                      inf_elem); // since we start indexing at 1
    std::sort(splitters.begin(), splitters.end());
    splitters_.resize(num_buckets, inf_elem);
    // PRINT_VAR(log_num_buckets);
    // PRINT_VAR(num_buckets);
    // PRINT_VECTOR(splitters_);
    for (std::size_t i = 1; i <= log_num_buckets; ++i) {
      std::size_t n = num_buckets / (1ull << i);
      const auto begin = splitters_.begin() + n;
      const auto end = begin + n;
      // PRINT_VAR(std::distance(splitters_.begin(), begin));
      // PRINT_VAR(std::distance(splitters_.begin(), end));
      std::generate(begin, end, [&, base = 1ull << (i - 1)]() mutable {
        const auto res =
            (base - 1) < splitters.size() ? splitters[base - 1] : inf_elem;
        base += 1ull << i;
        return res;
      });
      // PRINT_VECTOR(splitters_);
    }
    // PRINT_VECTOR(splitters_);
  }

  template <typename T> std::size_t compute_bucket(const T& elem) const {
    std::size_t bucket_idx = 1;
    for (std::size_t i = 0; i < log_num_buckets; ++i) {
      bucket_idx = 2 * bucket_idx + (elem > splitters_[bucket_idx]);
    }
    bucket_idx = bucket_idx - num_buckets;
    return bucket_idx;
  }
  template <class iterator, class Yield>
  void classifyUnrolled(iterator begin, const iterator end,
                        Yield&& yield) const {
    constexpr const std::size_t kNumBuckets = 1l << log_num_buckets;
    constexpr const int kUnroll = 15;

    std::size_t b[kUnroll];
    for (auto cutoff = end - kUnroll; begin <= cutoff; begin += kUnroll) {
      for (int i = 0; i < kUnroll; ++i)
        b[i] = 1;

      for (int l = 0; l < log_num_buckets; ++l)
        for (int i = 0; i < kUnroll; ++i)
          b[i] = 2 * b[i] + splitters_[b[i]] < begin[i];

      for (int i = 0; i < kUnroll; ++i)
        yield(b[i] - kNumBuckets, begin + i);
    }

    for (; begin != end; ++begin) {
      std::size_t b = 1;
      for (int l = 0; l < log_num_buckets; ++l)
        b = 2 * b + splitters_[b] > *begin;
      yield(b - kNumBuckets, begin);
    }
  }

private:
  Container splitters_;
  static constexpr std::size_t num_buckets = 1ull << log_num_buckets;
};

template <std::size_t log_num_buckets, typename Splitters, typename T,
          typename Iterator, typename Result>
hybridMST::non_init_vector<std::size_t>
classify(const Splitters& splitters, const T& inf_elem, Iterator begin,
         Iterator end, Result&& res) {
  const std::size_t n = std::distance(begin, end);
  Classifier_<log_num_buckets, std::vector<int>> classifier(splitters,
                                                            inf_elem);
  hybridMST::non_init_vector<std::size_t> res_vec(n);
  for (std::size_t i = 0; i < n; ++i) {
    res_vec[i] = classifier.compute_bucket(begin[i]);
  }
  return res_vec;
}

template <typename Splitters, typename T, typename Iterator, typename Result>
hybridMST::non_init_vector<std::size_t>
classify(std::size_t log_num_buckets, const Splitters& splitters,
         const T& inf_elem, Iterator begin, Iterator end, Result&& res) {
  switch (log_num_buckets) {
  case 1:
    return classify<1, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 2:
    return classify<2, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 3:
    return classify<3, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 4:
    return classify<4, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 5:
    return classify<5, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 6:
    return classify<6, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 7:
    return classify<7, Splitters, T, Iterator, Result>(
        splitters, inf_elem, begin, end, std::forward<Result>(res));
    break;
  case 8:
    return classify<8, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 9:
    return classify<9, Splitters, T, Iterator, Result>(splitters, inf_elem,
                                                       begin, end, res);
    break;
  case 10:
    classify<10, Splitters, T, Iterator, Result>(splitters, inf_elem, begin,
                                                 end, res);
    break;
  }
  return {};
}
