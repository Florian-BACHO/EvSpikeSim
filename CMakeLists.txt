cmake_minimum_required(VERSION 3.16.3)

project(evspikesim
  VERSION 1.0
  LANGUAGES C)

add_library(evspikesim STATIC
  src/cpu/fc_layer.c
  src/fc_layer_params.c
  src/network.c
  src/random.c
  src/spike_list.c)

target_include_directories (evspikesim PUBLIC inc)

install(TARGETS evspikesim DESTINATION lib)
install(FILES
  inc/fc_layer.h
  inc/fc_layer_params.h
  inc/layer_type.h
  inc/network.h
  inc/random.h
  inc/spike_list.h
  DESTINATION include/evspikesim)
