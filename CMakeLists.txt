cmake_minimum_required(VERSION 3.16)
project(hopfield_ode_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hopfield INTERFACE)
target_include_directories(hopfield INTERFACE include)

add_executable(hopfield_demo examples/hopfield_demo.cpp)
target_link_libraries(hopfield_demo PRIVATE hopfield)

add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver PRIVATE hopfield)

enable_testing()
add_test(NAME hopfield_solver_tests COMMAND test_solver)