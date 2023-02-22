//
// Created by Florian Bacho on 13/02/23.
//

#include <cstdlib>
#include <exception>
#include <evspikesim/Misc/JITCompiler.h>

using namespace EvSpikeSim;

static constexpr char compiler_cmd[] = "nvcc -ccbin g++ --x=cu -shared -Xcompiler -fPIC -o ";

void JITCompiler::compile(const std::string &source_file, const std::string &dlib_path) {
    std::string cmd = compiler_cmd;

    cmd += dlib_path + " " + source_file;
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error(std::string("Failed to compile source file: ") + source_file);
}