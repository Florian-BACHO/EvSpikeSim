cmake_minimum_required(VERSION 3.14)

file(STRINGS "../version.txt" VERSION_NUMBER)
message(STATUS "VERSION_NUMBER = ${VERSION_NUMBER}")

if (DEFINED BUILD_GPU)
    project(evspikesim_simulator
            VERSION ${VERSION_NUMBER}
            LANGUAGES CXX CUDA)

    find_package(CUDA REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    # Set default c++ compiler as base compiler of nvcc.
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -Wall -Wextra -O3")
else()
    project(evspikesim_simulator
            VERSION ${VERSION_NUMBER}
            LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

if (DEFINED BUILD_GPU)
    add_subdirectory(gpu)
else()
    add_subdirectory(cpu)
endif()
# Add tests but exclude them from install
add_subdirectory(tests EXCLUDE_FROM_ALL)