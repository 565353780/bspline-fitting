cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-std=c++17 -O0")
project(BSplineFitting)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(D_GLIBCXX_USE_CXX11_ABI 0)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

include_directories(${PROJECT_NAME} PRIVATE)

include_directories(./include)

file(GLOB_RECURSE SRC ./src/*.cpp)

add_executable(${PROJECT_NAME} ./src/main.cpp ${SRC})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)
