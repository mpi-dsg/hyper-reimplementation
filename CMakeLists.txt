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

# Unit / smoke tests for paper-disclosed algorithms
set(HYPER_LIB_SOURCES
        src/hyper_index.cpp
        src/leaf_node.cpp
        src/model_inner_node.cpp
        src/search_inner_node.cpp
        src/overflow_buffer.cpp
)

add_executable(test_paper_delete tests/test_paper_delete.cpp ${HYPER_LIB_SOURCES})
if(OpenMP_CXX_FOUND)
    target_compile_options(test_paper_delete PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_options(test_paper_delete PUBLIC ${OpenMP_CXX_FLAGS})
endif()
enable_testing()
add_test(NAME paper_delete COMMAND test_paper_delete)