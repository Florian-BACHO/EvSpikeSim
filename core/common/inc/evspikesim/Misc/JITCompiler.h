//
// Created by Florian Bacho on 13/02/23.
//

#pragma once

#include <filesystem>
#include <unordered_map>
#include <evspikesim/Misc/DynamicLibraryLoader.h>

namespace EvSpikeSim {
    class JITCompiler {
    public:
        JITCompiler(const std::string &dlib_dir);

        DynamicLibraryLoader &operator()(const std::string &source_file);

    private:
        void compile(const std::string &source_file, const std::string &dlib_path);

    private:
        std::filesystem::path dlib_dir;
        std::unordered_map<std::string, DynamicLibraryLoader> dlibs;
    };
}
