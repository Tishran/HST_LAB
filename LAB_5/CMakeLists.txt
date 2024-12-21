cmake_minimum_required(VERSION 3.28.3)
project(LAB_5 CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(LAB_5 main.cpp
        HDFUtils.cpp
        HDFUtils.h
        calculations_cpu.cpp
        calculations_cpu.h
        calculations_gpu.cu
        calculations_gpu.cuh)

set_target_properties(LAB_5 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(LAB_5 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
