cmake_minimum_required(VERSION 3.29)
project(LAB_4 CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(LAB_4 main.cpp
        HDFUtils.cpp
        HDFUtils.h
        calculations.cu
        calculations.cuh)

set_target_properties(LAB_4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(LAB_4 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
