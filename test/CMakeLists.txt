################################################################################
# Unit tests.
################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY unittests/)

set(TESTFILES 
  #test_alltoall.cpp 
  test_distributed_graph.cpp
  #test_local_contraction.cpp
  test_distributed_parent_array.cpp
  test_distributed_partitioning.cpp
  test_twolevel_alltoall.cpp
  #test_boruvka.cpp
  )
add_executable(tests test_main.cpp ${TESTFILES})
add_executable(tests_gbbs test_gbbs.cpp)
add_library(base_test_target INTERFACE)

target_include_directories(base_test_target
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/external
  ${PROJECT_SOURCE_DIR}/external/rmat-fork
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include/)
target_compile_options(base_test_target
  INTERFACE
  -O3
  -g
  -Wall
  -Wpedantic
  -Wextra)

target_link_libraries(base_test_target INTERFACE base_target Catch2::Catch2 timer graph_io)
target_link_libraries(tests PRIVATE base_test_target absl::flat_hash_map kadis)
target_link_libraries(tests_gbbs PRIVATE base_test_target gbbs)



