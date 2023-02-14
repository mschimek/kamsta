#pragma once
#include <vector>

namespace hybridMST {
template <typename T, typename Alloc = std::allocator<T>>
class default_init_allocator : public Alloc {
  using a_t = std::allocator_traits<Alloc>;

public:
  // obtain alloc<U> where U ≠ T
  template <typename U> struct rebind {
    using other =
        default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
  };
  // make inherited ctors visible
  using Alloc::Alloc;
  // default-construct objects
  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) { // 'placement new':
    ::new (static_cast<void*>(ptr)) U;
  }
  // construct with ctor arguments
  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<Alloc&>(*this), ptr,
                   std::forward<Args>(args)...);
  }
};

template <typename T>
std::vector<T>
vector_standard_alloc(const std::vector<T, default_init_allocator<T>>& vec) {
  std::vector<T> vec_std;
  for (auto elem : vec) {
    vec_std.push_back(std::move(elem));
  }
  return vec_std;
}
} // namespace hybridMST
