add_executable (demo_fc demo_fc.c)

target_link_libraries (demo_fc LINK_PUBLIC evspikesim)
target_link_libraries(demo_fc LINK_PUBLIC m) # Math library needs to be linked to any project using our simulator
