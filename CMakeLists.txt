cmake_minimum_required(VERSION 3.0.2)
project(lidar_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_definitions(-DCOMPILEDWITHC17)
message(STATUS "Using flag -std=c++17.")

add_definitions(-D__USE_SQUARE_BRACKETS_FOR_ELEMENT_ACCESS_OPERATOR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
  include
  thirdparty/
)

aux_source_directory(src/ SRC)

add_executable(unit_test unit_tests/main.cc ${SRC})

target_link_libraries(unit_test pthread)