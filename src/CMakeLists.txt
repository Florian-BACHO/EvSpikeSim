add_library(evspikesim
  cpu/fc_layer.c
  fc_layer_params.c
  network.c
  random.c
  spike_list.c)

target_include_directories (evspikesim PUBLIC ../inc)
