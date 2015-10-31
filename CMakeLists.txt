cmake_minimum_required(VERSION 3.0.0)
project(meded)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(USE_LIBCXX "Use libc++ for the C++ standard library" ON)

include(CTest)

add_subdirectory(../meta meta-libs EXCLUDE_FROM_ALL)

add_executable(stats src/stats.cpp)
target_link_libraries(stats cpptoml meta-util)

add_executable(active-l2r src/active_l2r.cpp)
target_link_libraries(active-l2r cpptoml meta-regression meta-classify)
