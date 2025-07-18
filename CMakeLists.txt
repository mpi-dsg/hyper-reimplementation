cmake_minimum_required(VERSION 3.27)
project(HyperCode)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mcx16 -latomic -mtune=native")

# Find and require OpenMP
find_package(OpenMP REQUIRED)


add_executable(HyperCode main.cpp
        include/piecewise_linear_model.hpp
        include/hyper_index.h
        include/leaf_node.h
        include/model_inner_node.h
        include/search_inner_node.h
        include/overflow_buffer.h
        src/hyper_index.cpp
        src/leaf_node.cpp
        src/model_inner_node.cpp
        src/search_inner_node.cpp
        include/common_defs.h
        src/overflow_buffer.cpp
        include/configuration_search.h
)

if(OpenMP_CXX_FOUND)
    target_compile_options(HyperCode PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_options(HyperCode PUBLIC ${OpenMP_CXX_FLAGS})
endif()