cmake_minimum_required(VERSION 2.8.12)
PROJECT(Cmake_test)

# 设置 C++ 标准 
set( CMAKE_CXX_FLAGS "-std=c++11")

# 添加源文件
file(GLOB SOURCES "src/*.cpp")

# 添加头文件路径
include_directories(include)


# 生成可执行文件
add_executable(gcd_demo ${SOURCES})
