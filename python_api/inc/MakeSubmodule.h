//
// Created by Florian Bacho on 28/01/23.
//

#pragma once

#define STR(str) #str
#define MAKE_SUBMODULE(mod, submod) object submod##_module(handle<>(borrowed(PyImport_AddModule(STR(mod.submod)))));\
scope().attr(STR(submod)) = submod##_module;\
scope submod##_scope = submod##_module;