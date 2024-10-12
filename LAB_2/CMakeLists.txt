cmake_minimum_required(VERSION 3.29)
project(LAB_2)

set(CMAKE_CXX_STANDARD 23)

add_executable(LAB_2 server/main.cpp
        server/TcpServer.cpp
        server/TcpServer.h
        server/calculations.cpp
        server/calculations.h
        server/Session.cpp
        server/Session.h
)
