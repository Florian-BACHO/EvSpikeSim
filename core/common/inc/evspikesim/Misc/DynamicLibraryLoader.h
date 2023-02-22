//
// Created by Florian Bacho on 13/02/23.
//

#pragma once

#include <dlfcn.h>
#include <string>

namespace EvSpikeSim {
    class DynamicLibraryLoader {
    public:
        DynamicLibraryLoader(const std::string &path, int flags = RTLD_NOW);

        ~DynamicLibraryLoader();

        void *operator()(const std::string &symbol);

    private:
        void *dynamic_lib;
    };
}
