cmake_minimum_required(VERSION 3.16)

project(dlframework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(playground src/main.cpp src/tensor.cpp)

target_include_directories(playground PRIVATE include)

target_compile_options(playground PRIVATE -Wall -Wextra -g)
