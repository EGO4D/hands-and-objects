cmake_minimum_required(VERSION 2.8)
project(CLUE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "../parrotscmake")

## Compiler settings

message(STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(NEWCXX_ENABLED_FLAG "-std=c++11 -stdlib=libc++ -Wall -O2")
    else()
        set(NEWCXX_ENABLED_FLAG "-std=c++11 -Wall -Wno-mismatched-tags -O2")
    endif()
elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    set(NEWCXX_ENABLED_FLAG "-std=c++11 -Wall -O2")
else()
    message(FATAL_ERROR, "Only GCC and clang is supported for now.")
endif()

set(CMAKE_CXX_FLAGS ${NEWCXX_ENABLED_FLAG})
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

enable_testing()

###################################
#
#  Dependencies
#
###################################

## Google Test

find_package(GTest REQUIRED)
if (GTEST_FOUND)
    include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
    message(STATUS "GTest Libraries: " ${GTEST_LIBRARIES})
endif()

# pthread library is required for Google Test to work properly
find_package(Threads REQUIRED)

###################################
#
#  Directories
#
###################################

include_directories(include)

set(TESTS tests)
set(EXECUTABLE_OUTPUT_PATH bin)

###################################
#
#  Unit testing
#
###################################

set(UNIT_TESTS
    test_misc
    test_predicates
    test_optional
    test_string_view
    test_stringex
    test_sformat
    test_stemplate
    test_type_name
    test_value_range
    test_array_view
    test_reindexed_view
    test_meta
    test_meta_seq
    test_textio
    test_include_all
)

foreach (tname ${UNIT_TESTS})
    add_executable(${tname} ${TESTS}/${tname}.cpp)
    target_link_libraries(${tname} ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(${tname} ${CMAKE_THREAD_LIBS_INIT})
    add_test(NAME ${tname} COMMAND ${tname})
endforeach()

set(THREADING_TESTS
    test_shared_mutex
    test_concurrent_counter
    test_concurrent_queue
    test_thread_pool
)

foreach(tname ${THREADING_TESTS})
    add_executable(${tname} ${TESTS}/${tname}.cpp)
    target_link_libraries(${tname} ${CMAKE_THREAD_LIBS_INIT})
    add_test(NAME ${tname} COMMAND ${tname})
endforeach()


###################################
#
#  Examples
#
###################################

set(EXAMPLES
    ex_stopwatch
    ex_timing
    ex_strings
)

set(THREAD_EXAMPLES
    ex_cccounter
    ex_ccqueue
    ex_threadpool
)

foreach (name ${EXAMPLES})
    add_executable(${name} examples/${name}.cpp)
endforeach()

foreach (name ${THREAD_EXAMPLES})
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} ${CMAKE_THREAD_LIBS_INIT})
endforeach()
