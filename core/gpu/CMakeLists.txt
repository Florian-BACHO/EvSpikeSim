message(STATUS "Compilation Target: GPU")

set(SRC_LIST
        ../common/src/Layers/Layer.cpp
        ../common/src/Layers/LayerDescriptor.cpp
        ../common/src/Spike.cpp
        ../common/src/SpikeArray.cpp
        ../common/src/SpikingNetwork.cpp
        src/Layers/Layer.cpp
        src/Layers/FCLayer.cpp
        src/SpikingNetwork.cpp)

set_source_files_properties(${SRC_LIST} PROPERTIES LANGUAGE CUDA)

add_library(evspikesim STATIC ${SRC_LIST})

target_link_libraries(evspikesim ${CUDA_LIBRARIES})

set_target_properties(evspikesim PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(evspikesim PUBLIC ../common/inc inc)

install(TARGETS evspikesim DESTINATION lib)
install(DIRECTORY inc/ DESTINATION include)
install(DIRECTORY ../common/inc/ DESTINATION include)