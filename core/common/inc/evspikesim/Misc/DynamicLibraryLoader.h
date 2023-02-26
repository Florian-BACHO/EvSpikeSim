//
// Created by Florian Bacho on 13/02/23.
//

#pragma once

#include <dlfcn.h>
#include <string>

namespace EvSpikeSim {
    /**
     * Loads dynamic libraries and fetches symbols.
     */
    class DynamicLibraryLoader {
    public:
        /**
         * Loads the given library with the given flags.
         * @param path Path to the dynamic library to load.
         * @param flags Flags of dlopen.
         * @throw std::runtime_error with the result of dlerror() if the library could not be loaded.
         */
        DynamicLibraryLoader(const std::string &path, int flags = RTLD_NOW);

        /**
         * Closes the loaded dynamic library
         */
        ~DynamicLibraryLoader();

        /**
         * Fetches the given symbol in the loaded dynamic library.
         * @param symbol Symbol to fetch.
         * @return The result of dlsym if succeeded.
         * @throw std::runtime_error with the result of dlerror() if the symbol could not be found in the loaded dynamic library.
         */
        void *operator()(const std::string &symbol);

    private:
        void *dynamic_lib; /**< The dynamic library loaded by dlopen. */
    };
}
