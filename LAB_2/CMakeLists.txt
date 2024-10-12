cmake_minimum_required(VERSION 3.29)
project(LAB_2)

set(CMAKE_CXX_STANDARD 23)
set( CMAKE_CXX_FLAGS_DEBUG  "")
set( CMAKE_CXX_FLAGS_RELEASE  "")
set( CMAKE_CXX_FLAGS  "-O0 -fopenmp")

add_executable(LAB_2 server/main.cpp
        server/TcpServer.cpp
        server/TcpServer.h
        server/calculations.cpp
        server/calculations.h
        server/Session.cpp
        server/Session.h
)
