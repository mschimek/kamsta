add_executable(comm_benchmarks comm_benchmarks.cpp)

OPTION(USE_EXPLICIT_INSTANTIATION "Build different TUs for variants and link them" OFF)
if (USE_EXPLICIT_INSTANTIATION)
  set(ALGO_EDGE_INSTANTIATIONS 
    algorithm_instantiations/algo_1.cpp
    algorithm_instantiations/algo_2.cpp
    algorithm_instantiations/algo_3.cpp
    algorithm_instantiations/algo_4.cpp
    algorithm_instantiations/algo_5.cpp
    algorithm_instantiations/algo_6.cpp
    algorithm_instantiations/algo_7.cpp
    algorithm_instantiations/algo_8.cpp)
else()
  set(ALGO_EDGE_INSTANTIATIONS )
endif()
configure_file(build_config.hpp.in build_config.hpp @ONLY)
message("ALGO_EDGE_INSTANTIATIONS: " ${ALGO_EDGE_INSTANTIATIONS})

add_executable(mst_benchmarks mst.cpp ${BACKWARD_ENABLE} ${ALGO_EDGE_INSTANTIATIONS})
target_include_directories(mst_benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

#set(STACK_DETAILS_DWARF ON)
#set(STACK_DETAILS_AUTO_DETECT OFF)
add_backward(mst_benchmarks)
target_include_directories(mst_benchmarks PRIVATE ${LIBDW_INCLUDE_DIR})
target_link_libraries(mst_benchmarks PRIVATE ${LIBDW_LIBRARY})
#target_compile_definitions(mst_benchmarks PUBLIC -DBACKWARD_HAS_DW=1)

add_executable(local_mst_benchmarks local_mst.cpp)
add_executable(growt_benchmarks growt.cpp)
add_executable(parlay_test parlay_test.cpp)

target_link_libraries(comm_benchmarks PRIVATE base_target timer)
target_link_libraries(mst_benchmarks PRIVATE base_target timer kadis absl::base absl::strings absl::flat_hash_map graphs)
target_link_libraries(local_mst_benchmarks PRIVATE base_target timer)
target_link_libraries(growt_benchmarks PRIVATE base_target)
target_link_libraries(parlay_test PRIVATE base_target)
