//
// Created by Florian Bacho on 22/01/23.
//

#include <stdexcept>
#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include <evspikesim/Misc/JITCompiler.h>

using namespace EvSpikeSim;

constexpr char mock_src_path[] = "/tmp/evspikesim_jit_mock_src.cpp";
constexpr char mock_lib_dir[] = "/tmp/evspikesim_jit_compiled";
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

class JitCompilerTest : public ::testing::Test {
public:
    JitCompilerTest() {
        write_src();
    }
};

TEST_F(JitCompilerTest, CompileTest) {
    JITCompiler compiler(mock_lib_dir);
    auto lib = compiler(mock_src_path);
    auto fct = reinterpret_cast<int (*)(const std::string &)>(lib("str_to_int"));

    EXPECT_EQ(fct("42"), 42);
}

TEST_F(JitCompilerTest, NoSuchFile) {
    JITCompiler compiler(mock_lib_dir);

    EXPECT_THROW(compiler("/no/such/file.cpp"), std::runtime_error);
}