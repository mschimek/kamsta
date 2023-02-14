#pragma once

#include "datastructures/distributed_array.hpp"
#include "definitions.hpp"
#include "mpi/allreduce.hpp"
#include "util/utils.hpp"

namespace hybridMST {
class OrgVertexRetrieval {
public:
  struct VertexOrgName {
    VId v;
    VId org_name;
    VertexOrgName() = default;
    VertexOrgName(VId v_, VId org_name_) : v{v_}, org_name{org_name_} {}
    friend std::ostream& operator<<(std::ostream& out,
                                    const VertexOrgName& v_org_name) {
      return out << "(" << v_org_name.v << "] = " << v_org_name.org_name;
    }
  };
  OrgVertexRetrieval(std::size_t n)
      : original_ids{
            n, [](std::size_t i, std::size_t begin) { return i + begin; }} {}

private:
  const mpi::MPIContext ctx_;
  DistributedArray<VId> original_ids;
};
} // namespace hybridMST
