if(NOT DEFINED NO_TEST OR NOT NO_TEST)
    find_package(GTest REQUIRED)

    enable_testing()
    if (DEFINED BUILD_GPU)
        include_directories(
                ${CMAKE_SOURCE_DIR}/evspikesim/common/inc
                ${CMAKE_SOURCE_DIR}/evspikesim/gpu/inc
                ${GTEST_INCLUDE_DIRS}
        )
    else()
        include_directories(
                ${CMAKE_SOURCE_DIR}/evspikesim/common/inc
                ${CMAKE_SOURCE_DIR}/evspikesim/cpu/inc
                ${GTEST_INCLUDE_DIRS}
        )
    endif()
    file (GLOB_RECURSE TEST_SOURCE_FILES "*.cpp")

    add_executable(tests ${TEST_SOURCE_FILES})

    if (DEFINED BUILD_GPU)
        set_source_files_properties(${TEST_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)
    endif ()
    target_link_libraries(tests evspikesim ${GTEST_BOTH_LIBRARIES})
endif()
