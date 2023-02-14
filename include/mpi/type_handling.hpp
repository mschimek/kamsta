#pragma once
// TODO add copyright claim for Florian Kurpicz since this is based on his work
#include "mpi.h"
#include <limits>

namespace hybridMST::mpi {
template <typename CXXType> class TypeMapper_ {
public:
  static constexpr MPI_Datatype get_mpi_datatype() { return MPI_BYTE; }
  static constexpr size_t get_factor() { return sizeof(CXXType); }
};

#define BUILTIN_TYPE_MAPPER(CXX_Type, MPI_Type)                                \
  template <> class TypeMapper_<CXX_Type> {                                    \
  public:                                                                      \
    static MPI_Datatype get_mpi_datatype() { return MPI_Type; }                \
    static constexpr size_t get_factor() { return 1u; }                        \
  };

BUILTIN_TYPE_MAPPER(bool, MPI_C_BOOL)
BUILTIN_TYPE_MAPPER(std::int8_t, MPI_INT8_T)
BUILTIN_TYPE_MAPPER(std::int16_t, MPI_INT16_T)
BUILTIN_TYPE_MAPPER(std::int32_t, MPI_INT32_T)
BUILTIN_TYPE_MAPPER(std::int64_t, MPI_INT64_T)
BUILTIN_TYPE_MAPPER(std::uint8_t, MPI_UINT8_T)
BUILTIN_TYPE_MAPPER(std::uint16_t, MPI_UINT16_T)
BUILTIN_TYPE_MAPPER(std::uint32_t, MPI_UINT32_T)
BUILTIN_TYPE_MAPPER(std::uint64_t, MPI_UINT64_T)
BUILTIN_TYPE_MAPPER(unsigned long long, MPI_UNSIGNED_LONG_LONG)
BUILTIN_TYPE_MAPPER(double, MPI_DOUBLE)
BUILTIN_TYPE_MAPPER(float, MPI_FLOAT)

template <typename CxxType> class TypeMapper {
public:
  TypeMapper() : is_builtin_(TypeMapper_<CxxType>::get_factor() == 1u) {
    if (is_builtin_) {
      mpi_datatype_ = TypeMapper_<CxxType>::get_mpi_datatype();
    } else {
      const int factor = TypeMapper_<CxxType>::get_factor();
      MPI_Type_contiguous(factor, MPI_BYTE, &mpi_datatype_);
      MPI_Type_commit(&mpi_datatype_);
    }
  }

  ~TypeMapper() {
    if (!is_builtin_)
      MPI_Type_free(&mpi_datatype_);
  }

  bool is_builtin() const { return is_builtin_; }
  MPI_Datatype get_mpi_datatype() const { return mpi_datatype_; }

private:
  MPI_Datatype mpi_datatype_;
  bool is_builtin_ = false;
};

inline MPI_Datatype build_blocks_type(const MPI_Datatype &base_type,
                                      int max_int, int nb_blocks) {
  MPI_Datatype block_type;
  MPI_Datatype blocks_type;
  MPI_Type_contiguous(max_int, base_type, &block_type);
  MPI_Type_commit(&block_type);
  MPI_Type_contiguous(nb_blocks, block_type, &blocks_type);
  MPI_Type_commit(&blocks_type);
  MPI_Type_free(&block_type);
  return blocks_type;
}

template <typename CxxType> MPI_Datatype get_big_type(std::size_t count) {
  MPI_Datatype result_type;
  const int max_int = std::numeric_limits<int>::max();
  const std::size_t nb_blocks = count / static_cast<std::size_t>(max_int);
  const std::size_t nb_remaining_elems =
      count % static_cast<std::size_t>(max_int);

  const TypeMapper<CxxType> tm;
  MPI_Datatype blocks_type =
      build_blocks_type(tm.get_mpi_datatype(), max_int, nb_blocks);
  if (nb_remaining_elems == 0u) {
    result_type = blocks_type;
  } else {
    MPI_Datatype remaining_elems_type;
    MPI_Type_contiguous(nb_remaining_elems, tm.get_mpi_datatype(),
                        &remaining_elems_type);
    MPI_Type_commit(&remaining_elems_type);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(tm.get_mpi_datatype(), &lb, &extent);
    const int block_count = 2;
    const int block_lengths[] = {1, 1};
    const MPI_Aint displ = static_cast<MPI_Aint>(nb_blocks) *
                           static_cast<MPI_Aint>(max_int) * extent;
    MPI_Aint displacements[2] = {0, displ};
    const MPI_Datatype types[2] = {blocks_type, remaining_elems_type};
    MPI_Type_create_struct(block_count, block_lengths, displacements, types,
                           &result_type);
    MPI_Type_commit(&result_type);
    MPI_Type_free(&blocks_type);
    MPI_Type_free(&remaining_elems_type);
  }
  return result_type;
}
} // namespace distMST::mpi
