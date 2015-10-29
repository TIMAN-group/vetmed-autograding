cmake_minimum_required(VERSION 3.0.0)
project(meded)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(USE_LIBCXX "Use libc++ for the C++ standard library" ON)

include(CTest)
include_directories(include/ ../meta/include)

add_subdirectory(../meta meta-libs EXCLUDE_FROM_ALL)

add_executable(stats src/stats.cpp)
target_link_libraries(stats cpptoml meta-util)
