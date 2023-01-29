include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
        evspikesim_tests
        Spike_test.cpp
        SpikeArray_test.cpp
        SpikeNetwork_test.cpp
        Layers/LayerDescriptor_test.cpp
        Layers/FCLayerDescriptor_test.cpp
        Layers/CPU/FCLayer_test.cpp
        Initializers/ConstantInitializer_test.cpp
        Initializers/UniformInitializer_test.cpp
        Initializers/NormalInitializer_test.cpp
        Misc/NDArray_test.cpp
)

target_include_directories (evspikesim_tests PUBLIC ../inc)

target_link_libraries(
        evspikesim_tests
        GTest::gtest_main
        evspikesim
)

include(GoogleTest)
gtest_discover_tests(evspikesim_tests)