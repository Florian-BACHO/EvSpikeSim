//
// Created by Florian Bacho on 13/02/23.
//

#include <iostream>
#include <evspikesim/Misc/DynamicLibraryLoader.h>
#include <stdexcept>

using namespace EvSpikeSim;

DynamicLibraryLoader::DynamicLibraryLoader(const std::string &path, int flags) :
        dynamic_lib(dlopen(path.c_str(), flags)) {
    if (dynamic_lib == nullptr)
        throw std::runtime_error(dlerror());
}

DynamicLibraryLoader::~DynamicLibraryLoader() {
    dlclose(dynamic_lib);
}

void *DynamicLibraryLoader::operator()(const std::string &symbol) {
    void *symbol_ptr = dlsym(dynamic_lib, symbol.c_str());

    if (symbol_ptr == nullptr)
        throw std::runtime_error(dlerror());
    return symbol_ptr;
}