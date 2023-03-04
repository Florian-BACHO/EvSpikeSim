#include <utility>

//
// Created by Florian Bacho on 13/02/23.
//

#include <iostream>
#include <evspikesim/Misc/JITCompiler.h>

using namespace EvSpikeSim;

JITCompiler::JITCompiler(const std::string &dlib_dir) : dlib_dir(dlib_dir) {
    std::filesystem::create_directories(dlib_dir);
}

DynamicLibraryLoader &JITCompiler::operator()(const std::string &source_file) {
    // Check if dynamic library is already loaded
    if (auto search = dlibs.find(source_file); search != dlibs.end())
        return search->second;

    // Dynamic library not loaded: compile
    auto dlib_path = dlib_dir / std::filesystem::path(source_file).stem();
    dlib_path.replace_extension(".so");

    compile(source_file, dlib_path.c_str());
    return dlibs.emplace(source_file, dlib_path.c_str()).first->second;
}