#pragma once

#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mpi/context.hpp"

//#include "build_config.h"

namespace hybridMST::internal {

inline std::string to_string_(const char* file, int line) {
  std::stringstream ss;
  ss << file << ":" << line << " ::";
  return ss.str();
}

template <typename... Args>
inline void print_(const std::string& name, const std::vector<Args...>& vec) {
  std::cout << "Content of vector: *" << name << "*\n\t";
  for (const auto& elem : vec)
    std::cout << elem << ",\n\t";
  std::cout << std::endl;
}

template <typename... Args>
inline void print_with_index_(const std::string& name,
                              const std::vector<Args...>& vec) {
  std::cout << "Content of vector: *" << name << "*\n\t";
  for (std::size_t i = 0; i < vec.size(); ++i) {
    const auto& elem = vec[i];
    std::cout << std::setw(10) << i << " " << elem << ",\n\t";
  }
  std::cout << std::endl;
}

template <typename Container>
inline void print_with_index_(const std::string& name, const Container& vec) {
  std::cout << "Content of vector: *" << name << "*\n\t";
  for (std::size_t i = 0; i < vec.size(); ++i) {
    const auto& elem = vec[i];
    std::cout << std::setw(10) << i << " " << elem << ",\n\t";
  }
  std::cout << std::endl;
}
// based
// https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
template <std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), std::string>::type
print_tuple(const std::tuple<Tp...>&) {
  return "";
}

template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), std::string>::type
                                     print_tuple(const std::tuple<Tp...>& t) {
  std::stringstream ss;
  ss << std::get<I>(t) << " ";
  ss << print_tuple<I + 1, Tp...>(t);
  return ss.str();
}

template <typename T1, typename T2>
std::string print_(const std::pair<T1, T2>& pair) {
  std::stringstream ss;
  ss << pair.first << " " << pair.second;
  return ss.str();
}

template <typename K, typename T>
inline void print_(const std::string& name,
                   const std::unordered_map<K, T>& map) {
  std::cout << "Content of map: *" << name << "*\n\t";
  for (const auto& [key, value] : map)
    std::cout << "key: " << key << " value: " << value << ",\n\t";
  std::cout << std::endl;
}

// template <typename K, typename T>
// inline void print_(const std::string& name,
//                    const google::dense_hash_map<K, T>& map) {
//   std::cout << "Content of map: *" << name << "*\n\t";
//   for (const auto& [key, value] : map)
//     std::cout << "key: " << key << " value: " << value << ",\n\t";
//   std::cout << std::endl;
// }

// template <typename T>
// inline void print_(const std::string& name,
//                    const google::dense_hash_set<T>& set) {
//   std::cout << "Content of set: *" << name << "*\n\t";
//   for (const auto& elem : set)
//     std::cout << elem << ",\n\t";
//   std::cout << std::endl;
// }

template <typename T>
inline void print_(const std::string& name, const std::set<T>& set) {
  std::cout << "Content of set: *" << name << "*\n\t";
  for (const auto& elem : set)
    std::cout << elem << ",\n\t";
  std::cout << std::endl;
}
template <typename T>
inline void print_(const std::string& name, const std::unordered_set<T>& set) {
  std::cout << "Content of unordered_set: *" << name << "*\n\t";
  for (const auto& elem : set)
    std::cout << elem << ",\n\t";
  std::cout << std::endl;
}

template <typename K, typename T>
inline void print_peid_vector(const std::string& name,
                              const std::unordered_map<K, T>& map) {
  std::cout << "Content of map: *" << name << "*\n\t";
  for (const auto& [pe, value] : map) {
    std::cout << "PE: " << pe << "\n";
    for (auto& elem : value)
      std::cout << "\t" << print_tuple(elem) << "\n";
  }
  std::cout << std::endl;
}

// template <>
// inline void print_peid_vector<PEID, std::vector<VId>>(const std::string&
// name,
//                               const std::unordered_map<PEID,
//                               std::vector<VId>>& map) {
//   std::cout << "Content of map: *" << name << "*\n\t";
//   for (const auto& [pe, value] : map) {
//     std::cout << "PE: " << pe << "\n";
//     for (auto& elem : value) std::cout << "\t" << elem << "\n";
//   }
//   std::cout << std::endl;
// }
//
// template <typename T>
// inline void print_peid_vector_non_tuple(const std::string& name,
//                               const std::unordered_map<PEID, std::vector<T>>&
//                               map) {
//   std::cout << "Content of map: *" << name << "*\n\t";
//   for (const auto& [pe, value] : map) {
//     std::cout << "PE: " << pe << "\n";
//     for (auto& elem : value) std::cout << "\t" << elem << "\n";
//   }
//   std::cout << std::endl;
// }

template <typename T> inline void print_(const std::string& name, const T& t) {
  std::cout << "Object: *" << name << "* (" << typeid(t).name() << ")\t" << t
            << std::endl;
}

template <typename T, typename U>
inline void print_(const std::string& name, const std::pair<T, U>& t) {
  std::cout << "Object: *" << name << "* (" << typeid(t).name() << ")\t("
            << t.first << ", " << t.second << ")" << std::endl;
}

} // namespace hybridMST::internal
#define LOCATION_INFO hybridMST::internal::to_string_(__FILE__, __LINE__)
#define PRINT_VECTOR(X) hybridMST::internal::print_(#X, X)
#define PRINT_CONTAINER_WITH_INDEX(X)                                          \
  hybridMST::internal::print_with_index_(#X, X)
#define PRINT_VECTOR_WITH_INDEX(X) hybridMST::internal::print_with_index_(#X, X)
#define PRINT_SET(X) hybridMST::internal::print_(#X, X)
#define PRINT_MAP(X) distMST::internal::print_(#X, X)
//#define PRINT_PEID_VECTOR(X) distMST::internal::print_peid_vector(#X, X)
//#define PRINT_PEID_VECTOR_NON_TUPLE(X)
// distMST::internal::print_peid_vector_non_tuple(#X, X)
#define PRINT_VAR(X) hybridMST::internal::print_(#X, X)
#define SEQ_EX(CTX, CODE) CTX.execute_in_order([&]() { CODE })

#define PRINT_WARNING(msg) {                                                    \
    std::stringstream ss;                                                       \
    hybridMST::mpi::MPIContext internal_ctx;                                               \
    ss << "WARNING at " << LOCATION_INFO << "\n\t" <<  msg;                     \
    if(internal_ctx.rank() == 0) {                                              \
      std::cout << ss.str() << std::endl;                                       \
    }                                                                           \
}

#define PRINT_WARNING_AND_ABORT(msg) {                                          \
    std::stringstream ss;                                                       \
    hybridMST::mpi::MPIContext internal_ctx;                                    \
    ss << "WARNING at " << LOCATION_INFO << "\n\t" <<  msg;                     \
    if(internal_ctx.rank() == 0) {                                              \
      std::cout << ss.str() << std::endl;                                       \
    }                                                                           \
    std::abort();                                                               \
}

#undef USE_ASSERTIONS_
#ifdef USE_ASSERTIONS_
#define MPI_ASSERT(ctx, condition, msg)                                        \
  {                                                                            \
    std::stringstream ss;                                                      \
    ctx.mpi_assert(condition, LOCATION_INFO, ((ss << msg), ss.str()));         \
  }

#define MPI_ASSERT_(condition, msg)                                            \
  {                                                                            \
    std::stringstream ss;                                                      \
    mpi::MPIContext internal_ctx;                                                       \
    internal_ctx.mpi_assert(condition, LOCATION_INFO, ((ss << msg), ss.str()));         \
  }
#else
#define MPI_ASSERT(ctx, condition, msg) ((void)0)
#define MPI_ASSERT_(condition, msg) ((void)0)
#endif

#define REORDERING_BARRIER asm volatile("" ::: "memory");
