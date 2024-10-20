cmake_minimum_required(VERSION 3.29)
project(LAB_3)

set(CMAKE_CXX_STANDARD 23)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(LAB_3 main.cpp
        calculations.cpp
        calculations.h
        HDFUtils.cpp
        HDFUtils.h
)

target_link_libraries(LAB_3 ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})