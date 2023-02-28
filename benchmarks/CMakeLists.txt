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
OPTION(USE_BACKWARD_CPP "Use Backward CPP" OFF)
if (USE_BACKWARD_CPP)
  set(BACKWARD_ENABLE_LIST ${BACKWARD_ENABLE})
else()
  set(BACKWARD_ENABLE_LIST "")
endif()

add_executable(mst_benchmarks mst.cpp ${BACKWARD_ENABLE_LIST} ${ALGO_EDGE_INSTANTIATIONS})
target_include_directories(mst_benchmarks PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if (USE_BACKWARD_CPP)
  add_backward(mst_benchmarks)
  target_include_directories(mst_benchmarks PRIVATE ${LIBDW_INCLUDE_DIR})
  target_link_libraries(mst_benchmarks PRIVATE ${LIBDW_LIBRARY})
endif()


target_link_libraries(mst_benchmarks PRIVATE base_target timer kadis graphs)
configure_file(build_config.hpp.in build_config.hpp @ONLY)
