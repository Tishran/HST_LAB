cmake_minimum_required(VERSION 3.29)
project(HST_LAB_1)

set(CMAKE_CXX_STANDARD 23)

add_executable(HST_LAB_1 server/main.cpp
        server/TcpServer.cpp
        server/TcpServer.h
        server/calculations.cpp
        server/calculations.h
        server/Session.cpp
        server/Session.h
)
