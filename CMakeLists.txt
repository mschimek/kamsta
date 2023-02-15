
cmake_minimum_required(VERSION 3.10) #not sure what we really need
project(HybridMST)

include(cmake_helpers/target_link_libraries_system.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RELEASE)
set(RQUICK_WITHOUT_OPENMP ON)

find_package(MPI REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(external/tlx)
add_subdirectory(external/graphs)
add_subdirectory(external/parlay)
add_subdirectory(external/abseil)
add_subdirectory(external/backward)

add_library(growt INTERFACE)
target_include_directories(growt SYSTEM INTERFACE external/growt/)

add_library(base_target INTERFACE)

target_compile_definitions(kadis PRIVATE RQUICK_WITHOUT_OPENMP)

list( APPEND WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wshadow"
    "-Wundef"
    "-Wunreachable-code"
    "-Wunused"
)


target_compile_options(base_target INTERFACE ${WARNING_FLAGS} -DPARLAY_OPENMP -O3 -march=native -mcx16 -Wno-psabi)
target_compile_definitions(base_target INTERFACE NDEBUG USE_ASSERTIONS)
target_compile_features(base_target INTERFACE cxx_std_17)

target_link_libraries(base_target INTERFACE MPI::MPI_CXX TBB::tbb OpenMP::OpenMP_CXX)
target_link_libraries_system(base_target INTERFACE parlay)
target_link_libraries_system(base_target INTERFACE kadis)

OPTION(USE_THREAD_SANITIZER "Compile with Thread Sanitizer enabled" OFF)
if (USE_THREAD_SANITIZER)
  target_compile_options(base_target INTERFACE -fsanitize=thread -g -fno-omit-frame-pointer)
  target_link_options(base_target INTERFACE -fsanitize=thread)
endif()
OPTION(USE_ADDRESS_SANITIZER "Compile with Thread Sanitizer enabled" OFF)
if (USE_ADDRESS_SANITIZER)
  message("USE_ADDRESS_SANITIZER: " ${USE_ADDRESS_SANITIZER})
  target_compile_options(base_target INTERFACE -fsanitize=address -g -fno-omit-frame-pointer)
  target_link_options(base_target INTERFACE -fsanitize=address)
endif()




target_link_libraries(base_target INTERFACE tlx atomic)
target_link_libraries(base_target INTERFACE growt)
target_include_directories(base_target SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/external/)
target_include_directories(base_target INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_library(timer ${CMAKE_CURRENT_SOURCE_DIR}/include/util/timer.cpp)
target_link_libraries(timer PRIVATE base_target)

add_subdirectory(benchmarks)



