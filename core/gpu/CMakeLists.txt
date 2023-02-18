message(STATUS "Compilation Target: GPU")

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
        src/Misc/GPUDevice.cpp
        src/Misc/JITCompiler.cpp
        )

set_source_files_properties(${SRC_LIST} PROPERTIES LANGUAGE CUDA)

add_library(evspikesim STATIC ${SRC_LIST})

target_link_libraries(evspikesim ${CUDA_LIBRARIES} stdc++fs)

set_target_properties(evspikesim PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LINKER_LANGUAGE CUDA)

target_include_directories(evspikesim PUBLIC ../common/inc inc)

install(TARGETS evspikesim DESTINATION lib)
install(DIRECTORY inc/ DESTINATION include)
install(DIRECTORY ../common/inc/ DESTINATION include)