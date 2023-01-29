cmake_minimum_required(VERSION 3.14)

file(STRINGS "version.txt" VERSION_NUMBER)


message(STATUS "VERSION_NUMBER = ${VERSION_NUMBER}")

project(evspikesim
        VERSION ${VERSION_NUMBER}
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src)
add_subdirectory(test)