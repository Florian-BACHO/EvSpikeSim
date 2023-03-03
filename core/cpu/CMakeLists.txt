cmake_minimum_required(VERSION 3.14)

message(STATUS "Compilation Target: CPU")

find_package(Threads REQUIRED)

set(SRC_LIST
        ../common/src/Layers/Layer.cpp
        ../common/src/Layers/FCLayer.cpp
        ../common/src/Layers/DefaultInferKernel.cpp
        ../common/src/Misc/DynamicLibraryLoader.cpp
        ../common/src/Misc/JITCompiler.cpp
        ../common/src/Spike.cpp
        ../common/src/SpikeArray.cpp
        ../common/src/SpikingNetwork.cpp
        src/Layers/Layer.cpp
        src/Misc/JITCompiler.cpp
        src/Misc/ThreadPool.cpp
        )

add_library(evspikesim STATIC ${SRC_LIST})

target_link_libraries(evspikesim pthread dl)

set_target_properties(evspikesim PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        POSITION_INDEPENDENT_CODE ON)

target_include_directories(evspikesim PUBLIC ../common/inc inc)

install(TARGETS evspikesim DESTINATION lib)
install(DIRECTORY inc/ DESTINATION include)
install(DIRECTORY ../common/inc/ DESTINATION include)