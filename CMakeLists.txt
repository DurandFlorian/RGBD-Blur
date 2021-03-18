cmake_minimum_required(VERSION 3.1)

# give a name to the project
project(depth-blur)

set(OpenCV_DIR C:/Users/flori/opencv/OpenCV-MinGW-Build-OpenCV-4.1.1-x64)

# find opencv
find_package(OpenCV REQUIRED)

# check opencv version
if(${OpenCV_VERSION} VERSION_LESS 2.0.0)
message(FATAL_ERROR “OpenCV version is not compatible : ${OpenCV_VERSION}”)
endif()

# compilation flags
set(CMAKE_CXX_FLAGS "-Wall -g -O2 -std=c++17")

# put the binary files in this directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# create a directory called 'output' in the project binary directory
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/output)

# the files to compile
add_executable(
    depth-blur 
    src/main.cpp
    )
target_link_libraries(depth-blur ${OpenCV_LIBS})
