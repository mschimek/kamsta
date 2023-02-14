#pragma once
#include <unistd.h>

#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

namespace hybridMST {
struct MemoryStats {
  //////////////////////////////////////////////////////////////////////////////
  // from
  // https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
  // process_mem_usage(double &, double &) - takes two doubles by reference,
  // attempts to read the system-dependent data for a process' virtual memory
  // size and resident set size, and return the results in KB.
  //
  // On failure, returns 0.0, 0.0

  void print(const std::string& desc) const {
    using std::ifstream;
    using std::ios_base;
    using std::string;

    if (is_disabled_)
      return;
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat", ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
        tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
        stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
        starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_bytes =
        sysconf(_SC_PAGE_SIZE); // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize;
    resident_set = rss * page_size_bytes;
    double rss_min;
    double rss_max;
    MPI_Allreduce(MPI_IN_PLACE, &rss_min, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &rss_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto vm_usage_ = static_cast<std::size_t>(vm_usage);
    static std::size_t first_vm_measurement = vm_usage_;
    if (rank == 0)
      std::cout << desc
                << ": \n\tVM:            " << static_cast<std::size_t>(vm_usage)
                << "(" << vm_usage << ") Bytes;  \n\tVM normalized: "
                << (vm_usage_ - first_vm_measurement)
                << "\n\tRSS: " << resident_set << "\n\tRSS_min: " << rss_min
                << "\n\tRSS_max: " << rss_max << std::endl;
  }
  void enable() { is_disabled_ = false; }
  void disable() { is_disabled_ = true; }
  bool is_disabled_ = false;
};

inline MemoryStats& memory_stats() {
  static MemoryStats stats;
  return stats;
}
} // namespace hybridMST
