add_library(evspikesim STATIC
        Spike.cpp
        SpikeArray.cpp
        SpikingNetwork.cpp
        Layers/LayerDescriptor.cpp
        Layers/Layer.cpp
        Layers/CPU/FCLayer.cpp)

set_target_properties(evspikesim PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories (evspikesim PUBLIC ../inc)