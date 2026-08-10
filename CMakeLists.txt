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
add_executable(test_paper_memory tests/test_paper_memory.cpp ${HYPER_LIB_SOURCES})
add_executable(test_paper_st_locks tests/test_paper_st_locks.cpp ${HYPER_LIB_SOURCES})
add_executable(test_insert_growth tests/test_insert_growth.cpp ${HYPER_LIB_SOURCES})
add_executable(test_paper_runtime tests/test_paper_runtime.cpp ${HYPER_LIB_SOURCES})
add_executable(test_paper_mt tests/test_paper_mt.cpp ${HYPER_LIB_SOURCES})
add_executable(test_planet_retry tests/test_planet_retry.cpp ${HYPER_LIB_SOURCES})
add_executable(test_leaf_bound_insert tests/test_leaf_bound_insert.cpp ${HYPER_LIB_SOURCES})
if(OpenMP_CXX_FOUND)
    foreach(t test_paper_delete test_paper_memory test_paper_st_locks test_insert_growth test_paper_runtime test_paper_mt test_planet_retry test_leaf_bound_insert)
        target_compile_options(${t} PUBLIC ${OpenMP_CXX_FLAGS})
        target_link_options(${t} PUBLIC ${OpenMP_CXX_FLAGS})
    endforeach()
endif()
find_package(Threads REQUIRED)
target_link_libraries(test_paper_mt PUBLIC Threads::Threads)
enable_testing()
add_test(NAME paper_delete COMMAND test_paper_delete)
add_test(NAME paper_memory COMMAND test_paper_memory)
add_test(NAME paper_st_locks COMMAND test_paper_st_locks)
add_test(NAME insert_growth COMMAND test_insert_growth)
add_test(NAME paper_runtime COMMAND test_paper_runtime)
add_test(NAME paper_mt COMMAND test_paper_mt)
add_test(NAME leaf_bound_insert COMMAND test_leaf_bound_insert)