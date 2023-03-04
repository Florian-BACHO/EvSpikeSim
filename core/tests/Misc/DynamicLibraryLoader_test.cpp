//
// Created by Florian Bacho on 13/02/23.
//

#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Misc/DynamicLibraryLoader.h>

using namespace EvSpikeSim;

constexpr char mock_src_path[] = "/tmp/evspikesim_mock_src.cpp";
constexpr char mock_lib_path[] = "/tmp/evspikesim_mock.so";
constexpr char mock_src[] = "#include <string>\n\n"
                            "extern \"C\" {\n"
                            "   int str_to_int(const std::string &i) {\n"
                            "       return std::stoi(i);\n"
                            "   }\n"
                            "}\n";

static void write_src() {
    std::ofstream out(mock_src_path);

    out << mock_src;
    out.close();
}

static void compile_mock_lib() {
    std::string command = "g++ -fPIC -shared -o ";

    command += mock_lib_path;
    command += " ";
    command += mock_src_path;
    if (std::system(command.c_str()))
        return;
}

class DynamicLibraryLoaderTest : public ::testing::Test {
public:
    DynamicLibraryLoaderTest() {
        write_src();
        compile_mock_lib();
    }
};

TEST_F(DynamicLibraryLoaderTest, LoadFunction) {
    DynamicLibraryLoader dlib(mock_lib_path);
    auto fct = reinterpret_cast<int (*)(const std::string &)>(dlib("str_to_int"));

    EXPECT_EQ(fct("42"), 42);
}

TEST_F(DynamicLibraryLoaderTest, NoSuchFile) {
    EXPECT_THROW(DynamicLibraryLoader("/no/such/file.so"), std::runtime_error);
}

TEST_F(DynamicLibraryLoaderTest, SymbolNotFound) {
    DynamicLibraryLoader dlib(mock_lib_path);

    EXPECT_THROW(dlib("no_such_symbol"), std::runtime_error);
}