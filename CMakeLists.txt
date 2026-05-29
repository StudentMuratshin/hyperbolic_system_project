cmake_minimum_required(VERSION 3.16)

project(HyperbolicShallowWaterComparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(shallow_water_compare
    src/main.cpp
    src/fluxes.cpp
    src/solver.cpp
)

target_include_directories(shallow_water_compare PRIVATE src)

if(MSVC)
    target_compile_options(shallow_water_compare PRIVATE /W4 /permissive-)
else()
    target_compile_options(shallow_water_compare PRIVATE -Wall -Wextra -pedantic)
endif()
