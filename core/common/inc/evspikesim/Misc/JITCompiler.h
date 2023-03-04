//
// Created by Florian Bacho on 13/02/23.
//

#pragma once

#include <filesystem>
#include <unordered_map>
#include <evspikesim/Misc/DynamicLibraryLoader.h>

namespace EvSpikeSim {
    /**
     * Just-In-Time (JIT) compiler. Compiles and load c++ source files as dynamic libraries.
     * In the CPU implementation, g++ is used. For the GPU implementation, nvcc is used with g++ as host compiler.
     */
    class JITCompiler {
    public:
        /**
         * Constructs a compiler with the given compile directory path.
         * @param dlib_dir The path to the directory where dynamic library are compiled.
         */
        JITCompiler(const std::string &dlib_dir);

        /**
         * Compiles and load the given source file. If the source file has already been compiled, JITCompiler skips
         * compilation and returns the pre-loaded dynamic library.
         * @param source_file Path to the source file to compile.
         * @return The loaded dynamic library.
         */
        DynamicLibraryLoader &operator()(const std::string &source_file);

    private:
        /**
         * Compiles source_file into the dlib_path dynamic library.
         * @param source_file  Path to the source file to compile.
         * @param dlib_path  Path to the output compiled dynamic library.
         */
        void compile(const std::string &source_file, const std::string &dlib_path);

    private:
        std::filesystem::path dlib_dir; /**< Path to the directory of compiled dynamic libraries. */
        std::unordered_map<std::string, DynamicLibraryLoader> dlibs; /**< All the pre-loaded dynamic library. */
    };
}
