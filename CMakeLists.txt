cmake_minimum_required(VERSION 3.8)
project(coursework1)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(coursework1 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(coursework1 ${OpenCV_LIBS})